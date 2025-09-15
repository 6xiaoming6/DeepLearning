import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from MyDataset import MyDataset
from UNet import UNet
from torchvision.utils import save_image

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'training in {device}')


weight_save_path = './params/u_net.pth'
data_path = r'/CNN/UNet/dataset/VOC2012'
save_image_path = 'result_images'

batch_size = 4
epochs = 100

if __name__ ==  "__main__":
    data_loader = DataLoader(dataset=MyDataset(data_path), batch_size=batch_size, shuffle=True)
    net = UNet().to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_func = nn.BCELoss()

    for epoch in range(epochs):
        for i, (img, segmentation) in enumerate(data_loader):
            img, segmentation = img.to(device), segmentation.to(device)
            #计算模型输出
            output = net(img)
            #计算损失
            loss = loss_func(output, segmentation)
            #梯度归零
            optimizer.zero_grad()
            #根据损失计算每个参数的梯度
            loss.backward()
            #根据梯度更新参数
            optimizer.step()
            print(f'{epoch}/{epochs}, i: {i}, train_loss=============>: {loss.item()}')
            if(i % 5 == 0):
               save_image(torch.stack([img[0], output[0], segmentation[0]], dim=0), os.path.join(save_image_path, f'img{epoch}-{i}.png'))

        torch.save(net.state_dict(), weight_save_path)


