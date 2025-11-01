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

x, y = data[:, :-6], data[:, -6:]

BATCH_SIZE = 16
LR = 0.0001
EPOCH = 200


def get_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

    # 针对每个标签创建 DataLoader
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

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


class CNN_ResNet(nn.Module):
    def __init__(self):
        super(CNN_ResNet, self).__init__()

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


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_data(x, y)
    model = CNN_ResNet()
    model = model.to(DEVICE)  # 将模型移动到适当的设备上

    # 定义损失函数
    loss_fn = torch.nn.MSELoss().to(DEVICE)
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

        total_losses = [0] * 6
        total_train_r2s = [0] * 6

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            losses = []
            for i in range(6):
                loss = loss_fn(outputs[:, i], targets[:, i])
                losses.append(loss)
                total_losses[i] += loss.item()
                total_train_r2s[i] += r2_score(targets.cpu().detach().numpy()[:, i],
                                               outputs.cpu().detach().numpy()[:, i])

            sum(losses).backward()
            optimizer.step()

        average_losses = [tl / len(train_loader) for tl in total_losses]
        average_train_r2s = [tr / len(train_loader) for tr in total_train_r2s]
        for i in range(6):
            print(
                f"Task {i + 1} - Epoch [{epoch + 1}/{EPOCH}], Train Loss: {average_losses[i]}, Train R2: {average_train_r2s[i]}")


        model.eval()  # 设置模型为评估模式

        with torch.no_grad():
            val_losses = [0] * 6
            total_val_r2s = [0] * 6

            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)

                for i in range(6):
                    loss = loss_fn(outputs[:, i], targets[:, i])
                    val_losses[i] += loss.item()
                    total_val_r2s[i] += r2_score(targets.cpu().numpy()[:, i], outputs.cpu().numpy()[:, i])

            average_val_losses = [vl / len(val_loader) for vl in val_losses]
            average_val_r2s = [vr / len(val_loader) for vr in total_val_r2s]
            for i in range(6):
                print(
                    f"Task {i + 1} - Epoch [{epoch + 1}/{EPOCH}], Validation Loss: {average_val_losses[i]}, Validation R2: {average_val_r2s[i]}")

