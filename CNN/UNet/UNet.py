from ConvBlock import ConvBlock
from DownSample import DownSample
from UpSample import UpSample

import torch.nn as nn
import torch as torch
import torch.optim as optim
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        #收缩路径各层(顺序从上到下)
        self.contract_cov1 = ConvBlock(input_channels=3, output_channels=64)
        self.down1 = DownSample(input_channels=64, output_channels=64)

        self.contract_cov2 = ConvBlock(input_channels=64, output_channels=128)
        self.down2 = DownSample(input_channels=128, output_channels=128)

        self.contract_cov3 = ConvBlock(input_channels=128, output_channels=256)
        self.down3 = DownSample(input_channels=256, output_channels=256)

        self.contract_cov4 = ConvBlock(input_channels=256, output_channels=512)
        self.down4 = DownSample(input_channels=512, output_channels=512)

        self.contract_cov5 = ConvBlock(input_channels=512, output_channels=1024)
        self.down5 = DownSample(input_channels=512, output_channels=1024)
        #拓展路径各层(顺序从下到上)
        self.up1 = UpSample(input_channels=1024, output_channels=512)
        self.expand_cov1 = ConvBlock(input_channels=1024, output_channels=512)

        self.up2 = UpSample(input_channels=512, output_channels=256)
        self.expand_cov2 = ConvBlock(input_channels=512, output_channels=256)

        self.up3 = UpSample(input_channels=256, output_channels=128)
        self.expand_cov3 = ConvBlock(input_channels=256, output_channels=128)

        self.up4 = UpSample(input_channels=128, output_channels=64)
        self.expand_cov4 = ConvBlock(input_channels=128, output_channels=64)
        #最后的输出层
        self.out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1)
        self.th = nn.Sigmoid()

    def forward(self, x):
        feature_map1 = self.contract_cov1(x)
        feature_map2 = self.contract_cov2(self.down1(feature_map1))
        feature_map3 = self.contract_cov3(self.down2(feature_map2))
        feature_map4 = self.contract_cov4(self.down3(feature_map3))
        feature_map5 = self.contract_cov5(self.down4(feature_map4))

        output = self.expand_cov1(self.up1(feature_map5,feature_map4))
        output = self.expand_cov2(self.up2(output,feature_map3))
        output = self.expand_cov3(self.up3(output,feature_map2))
        output = self.expand_cov4(self.up4(output,feature_map1))
        #将通道数转为输出结果需要的通道数
        output = self.out(output)
        output = self.th(output)
        return output

if __name__ == '__main__':
    u_net = UNet()
    x = torch.randn(1, 3, 256, 256)
    print(x.shape)
    output = u_net(x)
    print(output.shape)
