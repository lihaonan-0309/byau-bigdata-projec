import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# 从CSV文件中读取数据
data = pd.read_csv(r'C:\Users\l1824\Desktop\new_lucas_cleaned.csv')

# 假设前420列为特征，倒数第六列至倒数第二列为多个标签
X = data.iloc[:, :420]
y = data.iloc[:, 420:]

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建偏最小二乘回归模型，设置组件数，例如设置为 10
pls = PLSRegression(n_components=10)

# 训练模型
pls.fit(X_train, y_train)

# 预测
y_pred = pls.predict(X_test)

# 总体性能评估
mse_total = mean_squared_error(y_test, y_pred)
r2_total = r2_score(y_test, y_pred)
print('整体测试的均方误差（MSE）为：', mse_total)
print('整体测试的R²分数为：', r2_total)

# 打印每个标签的MSE和R²
for i in range(y_test.shape[1]):
    mse_label = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    r2_label = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f'标签 {i+1} 的均方误差（MSE）为：{mse_label}')
    print(f'标签 {i+1} 的R²分数为：{r2_label}')

# 将模型保存到本地文件中
filename = 'pls_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(pls, file)


import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# 从CSV文件中读取数据
data = pd.read_csv(r'C:\Users\l1824\Desktop\new_lucas_cleaned.csv')

# 假设前420列为特征，倒数第六列至倒数第二列为多个标签
X = data.iloc[:, :420]
y = data.iloc[:, 420:]

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建多输出K-近邻回归模型
model = KNeighborsRegressor()
multi_output_model = MultiOutputRegressor(model)

# 训练多输出回归模型
multi_output_model.fit(X_train, y_train)

# 将模型保存到本地文件中
filename = 'knn_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(multi_output_model, file)

# 使用测试数据预测结果
y_pred = multi_output_model.predict(X_test)

# 计算均方误差（MSE）
mse = mean_squared_error(y_test, y_pred)

# 计算R²分数
r2 = r2_score(y_test, y_pred)

print('测试的均方误差（MSE）为：', mse)
print('测试的R²分数为：', r2)

# 打印每个标签的MSE和R²
for i in range(y_test.shape[1]):
    mse_label = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    r2_label = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f'标签 {i+1} 的均方误差（MSE）为：{mse_label}')
    print(f'标签 {i+1} 的R²分数为：{r2_label}')


import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# 从CSV文件中读取数据
data = pd.read_csv(r'C:\Users\l1824\Desktop\new_lucas_cleaned.csv')

# 假设前420列为特征，倒数第六列至倒数第二列为多个标签
X = data.iloc[:, :420]
y = data.iloc[:, 420:]

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建梯度提升回归模型
gradient_boosting_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
multi_output_gb = MultiOutputRegressor(gradient_boosting_model)

# 训练多输出回归模型
multi_output_gb.fit(X_train, y_train)

# 将模型保存到本地文件中
filename = 'gb_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(multi_output_gb, file)

# 使用测试数据预测结果
y_pred = multi_output_gb.predict(X_test)

# 计算整体均方误差（MSE）
mse = mean_squared_error(y_test, y_pred)
# 计算整体R²分数
r2 = r2_score(y_test, y_pred)

print('测试的均方误差（MSE）为：', mse)
print('测试的R²分数为：', r2)

# 打印每个标签的MSE和R²
for i in range(y_test.shape[1]):
    mse_label = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    r2_label = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f'标签 {i+1} 的均方误差（MSE）为：{mse_label}')
    print(f'标签 {i+1} 的R²分数为：{r2_label}')


import pandas as pd
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# 从CSV文件中读取数据
data = pd.read_csv(r'C:\Users\l1824\Desktop\new_lucas_cleaned.csv')

# 假设前420列为特征，倒数第六列至倒数第二列为多个标签
X = data.iloc[:, :420]
y = data.iloc[:, 420:]

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建XGBoost回归模型
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
multi_output_xgb = MultiOutputRegressor(xgb_model)

# 训练多输出回归模型
multi_output_xgb.fit(X_train, y_train)

# 将模型保存到本地文件中
filename = 'xgb_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(multi_output_xgb, file)

# 使用测试数据预测结果
y_pred = multi_output_xgb.predict(X_test)

# 计算整体均方误差（MSE）
mse = mean_squared_error(y_test, y_pred)
# 计算整体R²分数
r2 = r2_score(y_test, y_pred)

print('测试的均方误差（MSE）为：', mse)
print('测试的R²分数为：', r2)

# 打印每个标签的MSE和R²
for i in range(y_test.shape[1]):
    mse_label = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    r2_label = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f'标签 {i+1} 的均方误差（MSE）为：{mse_label}')
    print(f'标签 {i+1} 的R²分数为：{r2_label}')

