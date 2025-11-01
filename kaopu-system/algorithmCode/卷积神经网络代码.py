import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = 'new_lucas_cleaned.csv'

data = np.array(pd.read_csv(path), dtype=np.float32)

x, y = data[:, :-6], data[:, -6]

BATCH_SIZE = 16
LR = 0.0001
EPOCH = 200


def get_data(x, y):
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

    # 创建训练集 DataLoader
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 创建验证集 DataLoader
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 创建测试集 DataLoader
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(64)  # 添加Batch Normalization层
        self.conv2 = nn.Conv1d(64, 128, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm1d(128)  # 添加Batch Normalization层
        self.conv3 = nn.Conv1d(128, 256, kernel_size=9, padding=4)
        self.bn3 = nn.BatchNorm1d(256)  # 添加Batch Normalization层
        self.conv4 = nn.Conv1d(256, 512, kernel_size=9, padding=4)
        self.bn4 = nn.BatchNorm1d(512)  # 添加Batch Normalization层

        # Max pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 26, 420)
        self.fc2 = nn.Linear(420, 1)

        # Flatten Layer
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加通道维度

        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_data(x, y)
    model = CNN()
    model = model.to(DEVICE)  # 将模型移动到适当的设备上

    # 定义损失函数
    loss = torch.nn.MSELoss().to(DEVICE)
    # 定义优化器
    optimizer = optim.AdamW(params=model.parameters(), lr=LR, weight_decay=5e-4)

    print('begin train on {}!'.format(DEVICE))

    best_val_loss = float('inf')
    early_stopping_count = 0

    train_loss_list = []
    val_loss_list = []
    train_r2_list = []
    val_r2_list = []

    for epoch in range(EPOCH):
        model.train()

        total_loss = 0
        total_train_r2 = 0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            batch_loss = loss(outputs, targets.unsqueeze(1))
            batch_loss.backward()

            optimizer.step()

            total_loss += batch_loss.item()
            total_train_r2 += r2_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())

        average_loss = total_loss / len(train_loader)
        average_train_r2 = total_train_r2 / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCH}], Average Loss: {average_loss}, Train R2: {average_train_r2}")

        train_loss_list.append(average_loss)
        train_r2_list.append(average_train_r2)

        # 在验证集上评估模型并进行早停
        model.eval()  # 设置模型为评估模式

        with torch.no_grad():
            val_losses = []
            total_val_r2 = 0

            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                outputs = model(inputs)

                batch_loss = loss(outputs, targets.unsqueeze(1))

                val_losses.append(batch_loss.item())
                total_val_r2 += r2_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())

            average_val_loss = np.mean(val_losses)
            average_val_r2 = total_val_r2 / len(val_loader)
            print(f"Epoch [{epoch + 1}/{EPOCH}], Validation Loss: {average_val_loss}, Validation R2: {average_val_r2}")

            val_loss_list.append(average_val_loss)
            val_r2_list.append(average_val_r2)

            if average_val_loss < best_val_loss:
                early_stopping_count = 0
                best_val_loss = average_val_loss
            else:
                early_stopping_count += 1

            if early_stopping_count >= 3:
                # 学习率衰减为之前的一半
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                    print(f"Learning rate decayed to: {param_group['lr']}")

                early_stopping_count = 0

            if early_stopping_count >= 10:
                print("Early stopping")
                break

            # 实时保存训练结果和验证结果到本地文件
            with open('PH.log', 'a+') as f:
                f.write(f'{average_loss},{average_val_loss},{average_train_r2},{average_val_r2}\n')
