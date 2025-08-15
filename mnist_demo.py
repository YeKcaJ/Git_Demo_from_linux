import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
# 1. 准备数据（自动下载）
transform = transforms.ToTensor() 
train_data = datasets.MNIST('data', train=True, download=True, transform=transform) 
train_loader = DataLoader(train_data, batch_size=32,shuffle=True)
# 2. 超简单模型
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 10) # 输入784像素，输出10个数字类别 
)
# 3. 训练（仅1个epoch）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model.train() 
for images, labels in train_loader:
    outputs = model(images)
    loss = nn.functional.cross_entropy(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Loss: {loss.item():.4f}')
