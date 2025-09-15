import torch.nn as nn
import torch.nn.functional as F

class DownSample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3,stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Dropout2d(0.25),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)