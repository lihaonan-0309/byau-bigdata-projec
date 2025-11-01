import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import r2_score
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


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_data(x, y)
    model = CNN_ResNet()
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
                torch.save(model.state_dict(), 'ph_model.pth')
                print("Best model saved to best_model_state_dict.pth")
            else:
                early_stopping_count += 1

            if early_stopping_count >= 5:
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
