import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Model import Model

batch_size = 64
epochs = 1000
transform = transforms.ToTensor()
device = 'cuda:0'
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print("使用设备:", device)


train_data = datasets.CIFAR10(root='./Dataset', train=True, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)


model = Model().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#开始训练
for epoch in range(epochs):
    i = 0
    print(f'{epoch}/{epochs}训练开始：')
    for data in train_loader:
        i = i + 1
        x, target = data
        x, target = x.to(device), target.to(device)
        output = model(x)
        loss = loss_fn(output, target)

        #参数优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'{epoch}/{epochs}：{i}==================>loss={loss.item()}')

    torch.save(model.state_dict(), './Weights/cifar10.pth')


