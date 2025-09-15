import torch.nn as nn
import torch.nn.functional as F
import torch


class UpSample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1)

    def forward(self, x, feature_map):
        #插值实现高宽翻倍
        upsample = F.interpolate(x, scale_factor=2, mode='bilinear')
        #使用1x1卷积核实现通道减半
        output = self.layer(upsample)
        #将压缩路径每层输出的feature_map和上采样的结果进行通道上的拼接
        return torch.concat([feature_map, output], dim=1)

