# Implement your UNet model here
import torch
from torch import nn

"""_summary_

    The contracting block of UNet.
    Each block doubles the number of input channels.
    Each block applies two 3x3 convolutions with ReLU (called DoubleConv),
    followed by a 2x2 MaxPool operation.
    The entire process is implemented in the Unet_Contracting_Block class.

    The expansive block of UNet.
    Each block first applies an up-convolution with a 2x2 kernel,
    then follows with a DoubleConv.
    The entire process is implemented in the Unet_Expansive_Block class.

    Note : The paper propose that the pooling layer should apply stride 2 for downsampling
"""


class DoubleConv(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(DoubleConv, self).__init__()

        # We add padding here for the purpose of concating data
        self.doubleConv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.doubleConv(x)


class Unet_Contracting_Block(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(Unet_Contracting_Block, self).__init__()

        self.input_channels = input_channels

        self.output_channels = output_channels

        self.doubleConv = DoubleConv(input_channels, output_channels)
        self.maxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.doubleConv(x)
        skip = x
        x = self.maxPool2d(x)
        return x, skip


class Unet_Expansive_Block(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(Unet_Expansive_Block, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.doubleConv = DoubleConv(input_channels, output_channels)
        # Set stride = 2 to double the size of input
        self.upConv2d = nn.ConvTranspose2d(
            input_channels, output_channels, kernel_size=2, stride=2
        )

    def forward(self, x, skip):
        # Data size (batch, channels, heights, widths)
        x = self.upConv2d(x)
        concat_input = torch.cat((x, skip), dim=1)
        x = self.doubleConv(concat_input)
        return x


class Unet(nn.Module):

    def __init__(self, input_channels, output_classes, channels=[64, 128, 256, 512]):
        """_summary_

        Args:
            input_channels (Int): The number of channels of input images
            output_classes (Int): The desired output number of classes for the purpose of segmentation
            channels (List) : Store the output channel of each contracting block
        """
        super(Unet, self).__init__()

        self.input_channels = input_channels
        self.output_classes = output_classes

        # List of contracing blocks
        self.contracting_blocks = nn.ModuleList()
        self.expansive_blocks = nn.ModuleList()

        contracting_block_input_channels = input_channels

        for channel in channels:

            self.contracting_blocks.append(
                Unet_Contracting_Block(contracting_block_input_channels, channel)
            )

            contracting_block_input_channels = channel

        self.bottom = DoubleConv(channels[-1], channels[-1] * 2)

        for channel in reversed(channels):
            self.expansive_blocks.append(Unet_Expansive_Block(channel * 2, channel))

        # In binary segmentation , the output should be the probability of foreground and background
        # So we applied sigmoid function at last
        self.output = nn.Sequential(
            nn.Conv2d(channels[0], output_classes, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x):
        skips = []
        # Apply the contracting path
        for block in self.contracting_blocks:
            x, skip = block(x)
            skips.append(skip)

        x = self.bottom(x)

        # Apply the expansiveve path
        for i, block in enumerate(self.expansive_blocks):
            skip_feature = skips[-(i + 1)]
            x = block(x, skip_feature)

        x = self.output(x)

        return x
