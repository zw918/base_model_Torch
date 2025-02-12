# source：https://www.zhihu.com/question/317687116/answer/70735541690
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
import time
import warnings
from pathlib import Path

# 配置设备：若可用则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据变换：将图像转换为张量，并归一化到指定均值和标准差
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载数据集
train_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 打印数据集大小
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)

# 定义神经网络模型
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

# 创建模型并移动到指定设备
model = MyModule()
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
criterion.to(device)
optimizer = optim.SGD(params=model.parameters(), lr=0.01)

# 训练模型
epochs = 10
for epoch in range(epochs):
    print(f'----------第{epoch}次训练----------')
    idx = 0
    model.train()  # 设置模型为训练模式
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 优化模型权重：清零梯度、反向传播、更新参数
        optimizer.zero_grad()
        # 反向传播计算得到每个参数的梯度值loss.backward()
        loss.backward()
        # 通过梯度下降执行一步参数更新optimizer.step()
        optimizer.step()

        # 可选：每200个 batch 打印一次损失（此处注释掉）
        # if idx % 200 == 0:
        #     print(f'loss {idx} = {loss.item()}')
        idx += 1

    # 测试阶段
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        model.eval()  # 设置模型为评估模式
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            accuracy = (outputs.argmax(dim=1) == labels).sum()
            total_accuracy += accuracy

    print(f'total_accuracy: {total_accuracy / test_data_size * 100:.2f} %')

    # 保存模型（在最后一次训练后保存）
    if epoch == epochs - 1:
        torch.save(model, f'model_{epoch}.pth')
        print('模型已保存')
