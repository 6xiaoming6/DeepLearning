import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Model import Model

transform = transforms.ToTensor()
test_data = datasets.CIFAR10(root='./Dataset', train=False, download=True, transform=transform)

test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

model = Model()
model.load_state_dict(torch.load("./Weights/cifar10.pth"))

total_data_size = len(test_data)
print(f'the total size = {total_data_size}')
correct_num = 0

model.eval()
for data in test_loader:
    x, target = data
    output = model(x)
    output = torch.argmax(output, dim=1)
    temp = (output == target)
    correct_num += temp.sum()
    print(f'预测情况{temp.sum()}/64')
print(f'总体准确率为{correct_num * 100/total_data_size}%')