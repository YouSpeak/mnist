import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import random_split
import os

import logging
import argparse

# 配置日志记录
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建解析器对象
parser = argparse.ArgumentParser(description='命令行参数示例')
# 添加命令行参数
parser.add_argument('--load_checkpoint', action='store_true', help='标志参数')
parser.add_argument('--checkpoint_path', default="", type=str, help='标志参数')
# 解析命令行参数
args = parser.parse_args()

logging.info("load_checkpoint: %s", args.load_checkpoint)
logging.info("checkpoint_path: %s", args.checkpoint_path)

# 校验checkpoint是否可用
if args.load_checkpoint:
    if args.checkpoint_path == "":
        logging.info("input params checkpoint_path is none, pick lastest checkpoint!!!")
    else:
        if not os.path.exists(args.checkpoint_path):
            raise Exception("checkpoint path is not exist")


# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
    
def find_last_checkpoint(directory):
    # 获取目录中的所有文件
    files = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

    # 按照文件的更新时间进行排序
    sorted_files = sorted(files, key=lambda x: os.path.getmtime(x), reverse=True)
    
    if len(sorted_files) == 0:
        return ""

    # 获取最新的文件路径
    newest_file = sorted_files[0]
    return newest_file
    
    
# 设置超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 10
test_git = 1

# 加载MNIST数据集
dataset = MNIST(root='.', train=True, transform=ToTensor(), download=True)
train_dataset, val_dataset = random_split(dataset, [50000, 10000])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 创建模型并将其移动到设备上
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 检查是否存在之前保存的检查点，如果存储目录，按照更新时间获取最新的checkpoint。
checkpoint_pre_path = "checkpoint"
if not os.path.exists(checkpoint_pre_path):
    os.mkdir(checkpoint_pre_path)

# 判断命令是否执行继续训练
if args.load_checkpoint:
    checkpoint_path_tmp = find_last_checkpoint(checkpoint_pre_path)
    logging.info("checkpoint_path is %s", checkpoint_path_tmp)
    if checkpoint_path_tmp == "":
        raise Exception("search checkpoint path is not exist, please check")
    checkpoint = torch.load(checkpoint_path_tmp)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
else:
    start_epoch = 0


# 训练循环
for epoch in range(start_epoch, num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        # 将数据移动到设备上
        data = data.to(device)
        targets = targets.to(device)

        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # 保存检查点
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    checkpoint_path_epoch = checkpoint_pre_path + "/" + str(epoch) + "-" + "checkpoint.pth"
    torch.save(checkpoint, checkpoint_path_epoch)

    # 在验证集上进行评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in val_loader:
            # 将数据移动到设备上
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch [{epoch}/{num_epochs}], Validation Accuracy: {accuracy:.2f}%')