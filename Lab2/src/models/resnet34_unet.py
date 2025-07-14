from torch import nn
import torch

# Reference :https://ithelp.ithome.com.tw/m/articles/10333931
# Each encoder_block contains a conv2d,batch_normalization and relu
class Residual_Block(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(Residual_Block, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
        )

    def forward(self, x):
        return self.encoder(x)


class ResNet32_UNet(nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        pass
