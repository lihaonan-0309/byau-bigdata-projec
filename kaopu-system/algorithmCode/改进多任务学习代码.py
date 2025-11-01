import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = 'new_lucas_cleaned.csv'

data = np.array(pd.read_csv(path), dtype=np.float32)

x, y = data[:, :-6], data[:, -6:]

BATCH_SIZE = 16
LR = 0.0001
EPOCH = 200
n_tasks = 6
alpha = 0.1


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


class CNNTrain(nn.Module):
    def __init__(self, model):
        super(CNNTrain, self).__init__()
        self.model = model
        self.weights = torch.nn.Parameter(torch.ones(model.n_tasks).float())
        self.mse_loss = nn.MSELoss()

    def forward(self, x, ts):
        B, n_tasks = ts.shape[:2]
        ys = self.model(x)

        assert (ys.size()[1] == n_tasks)
        task_loss = []
        r2_scores = []  # 存储每个任务的R2分数
        for i in range(n_tasks):
            task_loss.append(self.mse_loss(ys[:, i, :], ts[:, i, :]))
            r2_scores.append(r2_score(ts[:, i, :].cpu().detach().numpy(), ys[:, i, :].cpu().detach().numpy()))

        task_loss = torch.stack(task_loss)

        return task_loss, r2_scores

    def get_last_shared_layer(self):
        return self.model.get_last_shared_layer()


class CNN(nn.Module):
    def __init__(self, n_tasks):
        super(CNN, self).__init__()

        self.n_tasks = n_tasks

        self.conv1 = nn.Conv1d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=9, padding=4)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=9, padding=4)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=9, padding=4)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv5_branches = nn.ModuleList([nn.Conv1d(512, 128, kernel_size=1, stride=1) for _ in range(self.n_tasks)])

        self.fc1_branches = nn.ModuleList([nn.Linear(128 * 26, 420) for _ in range(self.n_tasks)])
        self.fc2_branches = nn.ModuleList([nn.Linear(420, 1) for _ in range(self.n_tasks)])

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加通道维度

        x = torch.relu(self.conv1(x))
        x = self.pool(x)

        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.relu(self.conv3(x))
        x = self.pool(x)

        x = torch.relu(self.conv4(x))
        x = self.pool(x)

        outputs = []

        for i, conv5_branch in enumerate(self.conv5_branches):
            branch_x = torch.relu(conv5_branch(x))

            branch_x = self.flatten(branch_x)  # 使用 Flatten 层将张量展平

            branch_x = torch.relu(self.fc1_branches[i](branch_x))

            branch_x = torch.relu(self.fc2_branches[i](branch_x))

            outputs.append(branch_x)

        return torch.stack(outputs, dim=1)

    def get_last_shared_layer(self):
        return self.conv4


if __name__ == '__main__':

    train_loader, val_loader, test_loader = get_data(x, y)

    model = CNNTrain(CNN(n_tasks=n_tasks))

    model = model.to(DEVICE)

    # 定义优化器
    optimizer = optim.AdamW(params=model.parameters(), lr=LR, weight_decay=5e-4)

    print('begin train on {}!'.format(DEVICE))

    best_val_loss = float('inf')
    early_stopping_count = 0

    weights = []
    task_losses = []
    loss_ratios = []
    grad_norm_losses = []

    train_loss_list = []
    val_loss_list = []
    train_r2_list = []
    val_r2_list = []

    for epoch in range(EPOCH):
        model.train()

        total_train_r2 = [0] * 6

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # 形状转换为(batch_size, n_tasks, 1)
            targets = torch.unsqueeze(targets, dim=2)

            # optimizer.zero_grad()

            # 计算每个任务的损失函数 L_i(t)
            task_loss, task_r2 = model(inputs, targets)

            for j in range(6):
                total_train_r2[j] += task_r2[j]

            # 计算加权损失 w_i(t) * L_i(t)
            weighted_task_loss = torch.mul(model.weights, task_loss)

            # 如果 t=0，则初始化初始损失 L(0)
            if i == 0:
                if torch.cuda.is_available():
                    initial_task_loss = task_loss.data.cpu()
                else:
                    initial_task_loss = task_loss.data
                initial_task_loss = initial_task_loss.numpy()

            # 计算总体损失
            loss = torch.sum(weighted_task_loss)

            # 清空梯度
            optimizer.zero_grad()

            # 对整个权重集合进行反向传播，计算出每个 \nabla_W L_i(t)
            loss.backward(retain_graph=True)

            # 将 w_i(t) 的梯度设置为零，梯度将使用 GradNorm 损失进行更新
            model.weights.grad.data = model.weights.grad.data * 0.0

            # 获取共享权重层
            W = model.get_last_shared_layer()

            # 计算每个任务的梯度范数 G^{(i)}_w(t)
            norms = []
            for i in range(len(task_loss)):
                # 计算该任务损失函数相对于共享参数的梯度
                gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
                # 计算范数
                norms.append(torch.norm(torch.mul(model.weights[i], gygw[0])))
            norms = torch.stack(norms)

            # 计算逆训练速率 r_i(t)
            if torch.cuda.is_available():
                loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
            else:
                loss_ratio = task_loss.data.numpy() / initial_task_loss
            inverse_train_rate = loss_ratio / np.mean(loss_ratio)

            # 计算平均范数 \tilde{G}_w(t)
            if torch.cuda.is_available():
                mean_norm = np.mean(norms.data.cpu().numpy())
            else:
                mean_norm = np.mean(norms.data.numpy())

            # 计算 GradNorm 损失
            constant_term = torch.tensor(mean_norm * (inverse_train_rate ** alpha), requires_grad=False)
            if torch.cuda.is_available():
                constant_term = constant_term.cuda()
            grad_norm_loss = torch.sum(torch.abs(norms - constant_term))

            # 计算权重的梯度
            model.weights.grad = torch.autograd.grad(grad_norm_loss, model.weights)[0]

            # 使用优化器进行一步更新
            optimizer.step()

        # 重新归一化权重
        normalize_coeff = n_tasks / torch.sum(model.weights.data, dim=0)
        model.weights.data = model.weights.data * normalize_coeff

        # 记录数据
        if torch.cuda.is_available():
            task_losses.append(task_loss.data.cpu().numpy())
            loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
            weights.append(model.weights.data.cpu().numpy())
            grad_norm_losses.append(grad_norm_loss.data.cpu().numpy())
        else:
            task_losses.append(task_loss.data.numpy())
            loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
            weights.append(model.weights.data.numpy())
            grad_norm_losses.append(grad_norm_loss.data.numpy())

        average_train_r2 = [r / len(train_loader) for r in total_train_r2]

        if torch.cuda.is_available():
            print('Epoch[{}/{}]: loss_ratio={}, weights={}, task_loss={}, grad_norm_loss={}, R2={}'.format(
                epoch + 1, EPOCH, loss_ratios[-1], model.weights.data.cpu().numpy(), task_loss.data.cpu().numpy(),
                grad_norm_loss.data.cpu().numpy(), average_train_r2))
        else:
            print('Epoch[{}/{}]: loss_ratio={}, weights={}, task_loss={}, grad_norm_loss={}, R2={}'.format(
                epoch + 1, EPOCH, loss_ratios[-1], model.weights.data.numpy(), task_loss.data.numpy(),
                grad_norm_loss.data.numpy(), average_train_r2))

        train_loss_list.append(task_loss.data.cpu().numpy())
        train_r2_list.append(average_train_r2)

        # 在验证集上评估模型并进行早停
        model.eval()

        with torch.no_grad():
            val_losses = [0] * n_tasks
            total_val_r2 = [0] * n_tasks

            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                # 形状转换为(batch_size, n_tasks, 1)
                targets = torch.unsqueeze(targets, dim=2)

                val_loss, val_r2 = model(inputs, targets)

                for j in range(n_tasks):
                    val_losses[j] += val_loss[j].item()
                    total_val_r2[j] += val_r2[j]

            average_val_loss = [l / len(val_loader) for l in val_losses]
            average_val_r2 = [r / len(val_loader) for r in total_val_r2]

            print(f"Epoch [{epoch + 1}/{EPOCH}], Validation Loss: {average_val_loss}, Validation R2: {average_val_r2}")

            val_loss_list.append(average_val_loss)
            val_r2_list.append(average_val_r2)

            if np.mean(average_val_loss) < best_val_loss:
                early_stopping_count = 0
                best_val_loss = np.mean(average_val_loss)
            else:
                early_stopping_count += 1

            if early_stopping_count >= 3:
                # 学习率衰减为之前的一半
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                    print(f"Learning rate decayed to: {param_group['lr']}")

                early_stopping_count = 0

            if early_stopping_count >= 15:
                print("Early stopping")
                break

            # # 实时保存训练结果和验证结果到本地文件
            # with open('MTL_PH.log', 'a+') as f:
            #     f.write(f'{average_loss[0]},{average_val_loss[0]},{average_train_r2[0]},{average_val_r2[0]}\n')
            # with open('MTL_OC.log', 'a+') as f:
            #     f.write(f'{average_loss[1]},{average_val_loss[1]},{average_train_r2[1]},{average_val_r2[1]}\n')
            # with open('MTL_CaCO3.log', 'a+') as f:
            #     f.write(f'{average_loss[2]},{average_val_loss[2]},{average_train_r2[2]},{average_val_r2[2]}\n')
            # with open('MTL_N.log', 'a+') as f:
            #     f.write(f'{average_loss[3]},{average_val_loss[3]},{average_train_r2[3]},{average_val_r2[3]}\n')
            # with open('MTL_P.log', 'a+') as f:
            #     f.write(f'{average_loss[4]},{average_val_loss[4]},{average_train_r2[4]},{average_val_r2[4]}\n')
            # with open('MTL_K.log', 'a+') as f:
            #     f.write(f'{average_loss[5]},{average_val_loss[5]},{average_train_r2[5]},{average_val_r2[5]}\n')
            with open('MTL_all.log', 'a+') as f:
                f.write(
                    f'{np.mean(task_loss.data.cpu().numpy())},{np.mean(average_val_loss)},{np.mean(average_train_r2)},{np.mean(average_val_r2)}\n')
