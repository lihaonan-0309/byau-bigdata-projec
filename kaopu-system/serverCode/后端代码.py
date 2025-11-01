import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from prophet import Prophet
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)  # 添加shortcut连接
        out = F.relu(out)

        return out


class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionModule, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = torch.sigmoid(out)  # 注意力权重

        out = torch.mul(residual, out)  # 乘以注意力权重

        return out


class CNN_ResNet(nn.Module):
    def __init__(self):
        super(CNN_ResNet, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(64)

        self.resblock1 = ResBlock(64, 128, stride=2)
        self.attention1 = AttentionModule(128, 128)  # 添加注意力模块
        self.resblock2 = ResBlock(128, 256, stride=2)
        self.attention2 = AttentionModule(256, 256)  # 添加注意力模块
        self.resblock3 = ResBlock(256, 512, stride=2)
        self.attention3 = AttentionModule(512, 512)  # 添加注意力模块

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512 * 27, 420)
        self.fc2 = nn.Linear(420, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.resblock1(x)
        x = self.attention1(x)  # 使用注意力模块
        x = self.resblock2(x)
        x = self.attention2(x)  # 使用注意力模块
        x = self.resblock3(x)
        x = self.attention3(x)  # 使用注意力模块

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class mtl(nn.Module):
    def __init__(self):
        super(mtl, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.resblock1 = ResBlock(64, 128, stride=2)
        self.resblock2 = ResBlock(128, 256, stride=2)
        self.resblock3 = ResBlock(256, 512, stride=2)
        self.fc_common = nn.Linear(512 * 27, 420)

        # 为每个标签任务定义一个输出层
        self.task_outputs = nn.ModuleList([nn.Linear(420, 1) for _ in range(6)])

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc_common(x)
        x = F.relu(x)

        # 分别为每个任务计算输出
        outputs = [task(x) for task in self.task_outputs]
        return torch.cat(outputs, dim=1)  # 将所有任务的输出合并为一个张量


# with open('random_forest_regression_model.pkl', 'rb') as file:
#    rf_model = pickle.load(file)


with open('pls_regression_model.pkl', 'rb') as file:
    pls_model = pickle.load(file)

with open('knn_regression_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)

with open('xgb_regression_model.pkl', 'rb') as file:
    xgboost_model = pickle.load(file)

with open('gb_regression_model.pkl', 'rb') as file:
    gradientboost_model = pickle.load(file)

label_names = {420: 'pH.in.H2O', 421: 'OC', 422: 'CaCO3', 423: 'N', 424: 'P', 425: 'K'}


def predict_model(model, request):
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # 读取上传的文件
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format'})

        # 提取所有特征
        X = pd.DataFrame(data)

        predictions = np.round(np.abs(model.predict(X)), decimals=2)

        results_dict = {}
        for i, row in enumerate(predictions):
            for j, value in enumerate(row):
                label_name = label_names[j + 420]
                if label_name not in results_dict:
                    results_dict[label_name] = []
                # 确保所有数值转换为 Python 的 float 类型
                results_dict[label_name].append(float(value))

        return jsonify(results_dict)

    except Exception as e:
        return jsonify({'error': str(e)})


def get_chrome_driver():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')  # 解决DevToolsActivePort文件不存在的报错
    chrome_options.add_argument('--disable-gpu')  # 谷歌文档提到需要加上这个属性来规避bug
    chrome_options.add_argument('--ignore-certificate-errors')  # 忽略证书错误

    chrome_driver_path = '/usr/bin/chromedriver'
    driver = webdriver.Chrome(executable_path=chrome_driver_path, options=chrome_options)

    return driver


def scrape_zixuns_page():
    driver = get_chrome_driver()

    zixuns_list = []

    driver.get('http://www.agri.cn/zx/xxlb/hlj/')

    time.sleep(2)

    zixuns = driver.find_elements(By.XPATH, '//ul[@class="nxw_list_ul"]/li')

    for index in range(1, len(zixuns) + 1):
        div_xpath = f'//ul[@class="nxw_list_ul"]/li[{index}]'
        new = driver.find_element(By.XPATH, div_xpath)

        title = new.find_element(By.XPATH, './/p[1]/a').get_attribute('title')
        href = new.find_element(By.XPATH, './/p[1]/a').get_attribute('href')
        content = new.find_element(By.XPATH, './/p[2]').text
        post_time = new.find_element(By.XPATH, './/div[@class="con_date "]/span').text

        zixuns_list.append({"title": title, "href": href, "content": content, "post_time": post_time})

    driver.quit()

    return zixuns_list


def scrape_zhengces_page():
    driver = get_chrome_driver()

    zhengces_list = []

    driver.get('http://nynct.hlj.gov.cn/nynct/c115385/public_list.shtml')

    time.sleep(2)

    zhengces = driver.find_elements(By.XPATH, '//ul[@id="list"]/li')

    for index in range(1, len(zhengces) + 1):
        div_xpath = f'//*[@id="list"]/li[{index}]'
        policy = driver.find_element(By.XPATH, div_xpath)

        title = policy.find_element(By.XPATH, './/a').get_attribute('title')
        href = policy.find_element(By.XPATH, './/a').get_attribute('href')
        post_time = policy.find_element(By.XPATH, './/span').text
        zhengces_list.append({"title": title, "href": href, "post_time": post_time})

    driver.quit()

    return zhengces_list


def scrape_redus_page():
    driver = get_chrome_driver()

    driver.get('https://tianji.ymt.com/#/')

    time.sleep(2)

    redus = driver.find_elements(By.XPATH, '//div[@class="list_srcoll"]/div/div')

    redus_list = []

    for index in range(2, len(redus) + 1):
        div_xpath = f'//div[@class="list_srcoll"]/div/div[{index}]'
        heat = driver.find_element(By.XPATH, div_xpath)

        title = heat.find_element(By.XPATH, './/div[1]').text
        buyer = heat.find_element(By.XPATH, './/div[2]').text
        buyerheat = buyer.split(" ")[0]
        buyertrend = buyer.split(" ")[1]

        seller = heat.find_element(By.XPATH, './/div[3]').text
        sellerheat = seller.split(" ")[0]
        sellertrend = seller.split(" ")[1]

        redus_list.append({"title": title, "buyerheat": buyerheat, "buyertrend": buyertrend, "sellerheat": sellerheat,
                           "sellertrend": sellertrend})

    driver.quit()

    return redus_list


def scrape_jiages_page():
    driver = get_chrome_driver()

    data_list = []
    c = 1

    url = 'https://price.21food.cn/market/1015-p' + str(c) + '.html'

    while c <= 20:
        driver.get(url)
        time.sleep(2)

        logs = driver.find_elements(By.XPATH, '/html/body/div[2]/div[3]/div/div[2]/div[1]/div[2]/div[1]/ul/li')

        for index in range(1, len(logs) + 1):
            div_xpath = f'/html/body/div[2]/div[3]/div/div[2]/div[1]/div[2]/div[1]/ul/li[{index}]'
            log = driver.find_element(By.XPATH, div_xpath)

            mc = log.find_element(By.XPATH, './/table/tbody/tr/td[1]/a').text
            max_price = log.find_element(By.XPATH, './/table/tbody/tr/td[4]/span').text
            min_price = log.find_element(By.XPATH, './/table/tbody/tr/td[5]/span').text
            avg_price = log.find_element(By.XPATH, './/table/tbody/tr/td[6]/span').text
            date = log.find_element(By.XPATH, './/table/tbody/tr/td[7]/span').text

            data_list.append({"名称": mc, "最高价": max_price, "最低价": min_price, "平均价": avg_price, "日期": date})

        c += 1
        url = 'https://price.21food.cn/market/1015-p' + str(c) + '.html'

    driver.quit()

    return data_list


def scrape_200ri_page():
    driver = get_chrome_driver()

    driver.get('https://pfsc.agri.cn/#/priceExponent')

    time.sleep(2)

    logs = driver.find_elements(By.XPATH,
                                '//*[@id="app"]/div/div[2]/div[2]/div[2]/div[2]/div/div/div/div[3]/table/tbody/tr')

    data_list = []

    for index in range(1, len(logs) + 1):
        div_xpath = f'//*[@id="app"]/div/div[2]/div[2]/div[2]/div[2]/div/div/div/div[3]/table/tbody/tr[{index}]'
        log = driver.find_element(By.XPATH, div_xpath)

        post_time = log.find_element(By.XPATH, './/td[1]/div').text
        ncp = log.find_element(By.XPATH, './/td[2]/div').text
        clz = log.find_element(By.XPATH, './/td[3]/div').text
        ly = log.find_element(By.XPATH, './/td[4]/div').text

        data_list.append({"post_time": post_time, "ncp": ncp, "clz": clz,
                          "ly": ly})

    driver.quit()

    return data_list


def scrape_page():
    driver = get_chrome_driver()

    data_dict = {}
    c = 1

    url = 'https://price.21food.cn/market/1015-p' + str(c) + '.html'

    while c <= 20:
        driver.get(url)
        time.sleep(2)

        logs = driver.find_elements(By.XPATH, '/html/body/div[2]/div[3]/div/div[2]/div[1]/div[2]/div[1]/ul/li')

        for index in range(1, len(logs) + 1):
            div_xpath = f'/html/body/div[2]/div[3]/div/div[2]/div[1]/div[2]/div[1]/ul/li[{index}]'
            log = driver.find_element(By.XPATH, div_xpath)

            mc = log.find_element(By.XPATH, './/table/tbody/tr/td[1]/a').text
            max_price_str = log.find_element(By.XPATH, './/table/tbody/tr/td[4]/span').text
            min_price_str = log.find_element(By.XPATH, './/table/tbody/tr/td[5]/span').text
            avg_price_str = log.find_element(By.XPATH, './/table/tbody/tr/td[6]/span').text
            date = log.find_element(By.XPATH, './/table/tbody/tr/td[7]/span').text

            max_price = float(max_price_str.split('元')[0]) if '元' in max_price_str else 0.0
            min_price = float(min_price_str.split('元')[0]) if '元' in min_price_str else 0.0
            avg_price = float(avg_price_str.split('元')[0]) if '元' in avg_price_str else 0.0

            if mc not in data_dict:
                data_dict[mc] = []
            data_dict[mc].append((date, avg_price))

        c += 1
        url = 'https://price.21food.cn/market/1015-p' + str(c) + '.html'

    driver.quit()

    return data_dict


def forecast_prices(data_dict):
    forecast_data = {}
    for item_name, prices in data_dict.items():
        df = pd.DataFrame(prices, columns=['ds', 'y'])

        m = Prophet()
        m.fit(df)

        future = m.make_future_dataframe(periods=7)

        forecast = m.predict(future)

        forecast_data[item_name] = forecast[['ds', 'yhat']].tail(7).to_dict('records')

    return forecast_data


def df_to_records(df):
    return [row.to_dict() for index, row in df.iterrows()]


@app.route('/smooth', methods=['POST'])
def smooth_data():
    uploaded_file = request.files['file']
    input_df = pd.read_csv(uploaded_file)

    window_size = 3
    smoothed_df = input_df.rolling(window=window_size, min_periods=1).mean()

    # 返回行格式的JSON数据
    response = jsonify({
        'original_data': df_to_records(input_df),
        'processed_data': df_to_records(smoothed_df)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/derive', methods=['POST'])
def derive_data():
    uploaded_file = request.files['file']
    input_df = pd.read_csv(uploaded_file)

    # 填充缺失值为0
    input_df.fillna(0, inplace=True)

    derivative_df = input_df.diff()

    # 返回行格式的JSON数据
    response = jsonify({
        'original_data': df_to_records(input_df),
        'processed_data': df_to_records(derivative_df)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response



@app.route('/msc', methods=['POST'])
def msc_correct_data():
    uploaded_file = request.files['file']
    input_df = pd.read_csv(uploaded_file)

    # 假设光谱数据位于所有列中，如有必要请自行调整列范围
    spectral_data = input_df.copy()

    # 计算平均光谱
    mean_spectrum = spectral_data.mean(axis=0)

    # 初始化经过MSC处理的DataFrame
    msc_df = pd.DataFrame()

    # 对每个样本执行MSC
    for i in range(spectral_data.shape[0]):
        sample_spectrum = spectral_data.iloc[i, :]

        # MSC的线性回归拟合
        fit = np.polyfit(mean_spectrum, sample_spectrum, 1)
        slope = fit[0]
        intercept = fit[1]

        # 根据回归系数调整样本光谱
        corrected_spectrum = (sample_spectrum - intercept) / slope

        # 将校正后的光谱加入到MSC DataFrame
        msc_df = pd.concat([msc_df, pd.DataFrame(corrected_spectrum).T])

    # 重置DataFrame的索引，因为我们连续地concat了多个DataFrame
    msc_df.reset_index(drop=True, inplace=True)

    # 返回行格式的JSON数据
    response = jsonify({
        'original_data': df_to_records(input_df),
        'processed_data': df_to_records(msc_df)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/standardize', methods=['POST'])
def standardize_data():
    uploaded_file = request.files['file']
    input_df = pd.read_csv(uploaded_file)

    features_to_standardize = input_df.select_dtypes(include=['float64', 'int64'])
    standardized_features = (features_to_standardize - features_to_standardize.mean()) / features_to_standardize.std()

    standardized_df = pd.concat([input_df.select_dtypes(exclude=['float64', 'int64']), standardized_features], axis=1)

    # 返回行格式的JSON数据
    response = jsonify({
        'original_data': df_to_records(input_df),
        'processed_data': df_to_records(standardized_df)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/detrend', methods=['POST'])
def detrend_data():
    uploaded_file = request.files['file']
    input_df = pd.read_csv(uploaded_file)

    # 标准正态变化（标准化）
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(input_df)
    scaled_df = pd.DataFrame(scaled_data, columns=input_df.columns)

    # 使用一阶差分去趋势处理
    detrended_data = scaled_df.diff().dropna()

    # 返回JSON数据
    response = jsonify({
        'original_data': df_to_records(input_df),
        'processed_data': df_to_records(detrended_data)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/zixun', methods=['GET'])
def get_news():
    zixuns = scrape_zixuns_page()
    response = jsonify({"zixun": zixuns})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/zhengce', methods=['GET'])
def get_policies():
    zhengces = scrape_zhengces_page()
    response = jsonify({"zhengce": zhengces})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/jiage', methods=['GET'])
def get_agriculture_heats():
    data = scrape_jiages_page()
    response = jsonify({"jiage": data})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/redu', methods=['GET'])
def get_agriculture_price():
    data = scrape_redus_page()
    response = jsonify({"redu": data})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/200ri', methods=['GET'])
def get_agriculture_200ri():
    data = scrape_200ri_page()
    response = jsonify({"200ri": data})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/yuce', methods=['GET'])
def get_forecast():
    data_dict = scrape_page()
    forecast_data = forecast_prices(data_dict)
    response = jsonify(forecast_data)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_pls', methods=['POST'])
def predict_rf():
    response = predict_model(pls_model, request)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    response = predict_model(knn_model, request)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_xgboost', methods=['POST'])
def predict_xgboost():
    response = predict_model(xgboost_model, request)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_gradientboost', methods=['POST'])
def predict_gradientboost():
    response = predict_model(gradientboost_model, request)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/cnn_ph', methods=['POST'])
def predict_ph():
    if request.files:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        model_path = 'ph_model.pth'
        model = CNN_ResNet()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        data = data.values.astype(np.float32)
        x_predict = torch.from_numpy(data)

        with torch.no_grad():
            outputs = model(x_predict)
            predictions = outputs.cpu().numpy().tolist()

        response = jsonify({'ph': predictions})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    else:
        response = jsonify({'error': 'No file provided'}), 400
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


@app.route('/cnn_oc', methods=['POST'])
def predict_oc():
    if request.files:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        model_path = 'oc_model.pth'
        model = CNN_ResNet()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        data = data.values.astype(np.float32)
        x_predict = torch.from_numpy(data)

        with torch.no_grad():
            outputs = model(x_predict)
            predictions = outputs.cpu().numpy().tolist()

        response = jsonify({'oc': predictions})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    else:
        response = jsonify({'error': 'No file provided'}), 400
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


@app.route('/cnn_caco3', methods=['POST'])
def predict_caco3():
    if request.files:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        model_path = 'caco3_model.pth'
        model = CNN_ResNet()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        data = data.values.astype(np.float32)
        x_predict = torch.from_numpy(data)

        with torch.no_grad():
            outputs = model(x_predict)
            predictions = outputs.cpu().numpy().tolist()

        response = jsonify({'caco3': predictions})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    else:
        response = jsonify({'error': 'No file provided'}), 400
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


@app.route('/cnn_n', methods=['POST'])
def predict_n():
    if request.files:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        model_path = 'n_model.pth'
        model = CNN_ResNet()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        data = data.values.astype(np.float32)
        x_predict = torch.from_numpy(data)

        with torch.no_grad():
            outputs = model(x_predict)
            predictions = outputs.cpu().numpy().tolist()

        response = jsonify({'n': predictions})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    else:
        response = jsonify({'error': 'No file provided'}), 400
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


@app.route('/cnn_p', methods=['POST'])
def predict_p():
    if request.files:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        model_path = 'p_model.pth'
        model = CNN_ResNet()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        data = data.values.astype(np.float32)
        x_predict = torch.from_numpy(data)

        with torch.no_grad():
            outputs = model(x_predict)
            predictions = outputs.cpu().numpy().tolist()

        response = jsonify({'p': predictions})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    else:
        response = jsonify({'error': 'No file provided'}), 400
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


@app.route('/cnn_k', methods=['POST'])
def predict_k():
    if request.files:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        model_path = 'k_model.pth'
        model = CNN_ResNet()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        data = data.values.astype(np.float32)
        x_predict = torch.from_numpy(data)

        with torch.no_grad():
            outputs = model(x_predict)
            predictions = outputs.cpu().numpy().tolist()

        response = jsonify({'k': predictions})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    else:
        response = jsonify({'error': 'No file provided'}), 400
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


@app.route('/predict_all', methods=['POST'])
def predict_all():
    if request.files:
        file = request.files['file']
        # 支持 csv 和 Excel 文件
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        # 转换数据为模型所需格式
        data = data.values.astype(np.float32)
        x_predict = torch.from_numpy(data)

        # 加载模型并预测
        response_data = {}
        for label, model_path in [('ph', 'ph_model.pth'),
                                  ('oc', 'oc_model.pth'),
                                  ('caco3', 'caco3_model.pth'),
                                  ('n', 'n_model.pth'),
                                  ('p', 'p_model.pth'),
                                  ('k', 'k_model.pth')]:
            model = CNN_ResNet()
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            with torch.no_grad():
                outputs = model(x_predict)
                predictions = np.abs(outputs.cpu().numpy()).flatten().tolist()
            # 将所有预测值存储为列表
            if label in response_data:
                response_data[label].extend(predictions)
            else:
                response_data[label] = predictions

        # 返回所有预测结果
        response = jsonify(response_data)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    else:
        return jsonify({'error': 'No file provided'}), 400


@app.route('/mtl', methods=['POST'])
def predict_mtl():
    if request.files:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        model_path = 'mtl_model.pth'
        model = mtl()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        data = data.values.astype(np.float32)
        x_predict = torch.from_numpy(data)

        with torch.no_grad():
            outputs = model(x_predict)
            predictions = outputs.cpu().numpy().tolist()

        response = jsonify({'mtl': predictions})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    else:
        response = jsonify({'error': 'No file provided'}), 400
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=59687, debug=False)
