from torch import nn
import torch


class Unet_Expansive_Block(nn.Module):

    def __init__(self, input_channels, mid_channels, output_channels):
        super(Unet_Expansive_Block, self).__init__()

        # We should upconv input first to align the shape between input and shortcut

        self.doubleConv = DoubleConv(output_channels, output_channels)
        # Set stride = 2 to double the size of input
        self.upConv2d = nn.ConvTranspose2d(
            input_channels + mid_channels, output_channels, kernel_size=2, stride=2
        )

    def forward(self, x, skip):
        # Data size (batch, channels, heights, widths)
        x = torch.cat((x, skip), dim=1)
        x = self.upConv2d(x)

        x = self.doubleConv(x)
        return x


class DoubleConv(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(DoubleConv, self).__init__()

        # We add padding here for the purpose of concating data
        self.doubleConv = nn.Sequential(
            nn.Conv2d(
                input_channels, output_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels, output_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.doubleConv(x)


# Reference :https://www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch
# Each encoder_block contains a conv2d,batch_normalization and relu

# /2 sign indicates that the shape of input should be halved
# OutSize=floor((InSize+2*padding -kernel_size / stride + 1)
# Assumer that padding = 1, kernel_size =3 then,
# If stride =1 , then The OutSize is equal to InSize
# stride =2 ,then The OutSize is equal to half of InSize
# That's the reason that why we set stride as a variable instead of a constant


class Residual_Block(nn.Module):

    def __init__(self, input_channels, output_channels, stride=1, downsample=None):

        super(Residual_Block, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        # Since BatchNorm is already provided bias , we dont need to do it on Conv2d anymore
        self.doubleConv = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
        )
        # y = F(x) + x , we called x shortcut here
        # We need downsample for the purpose of aligning channel of shortcut
        # EX : According to the resnet34 papper , from conv2_x to conv3_x , the number of channels will convert from 64 into 128
        # In the meanwhile , the channel of F(x) is 128 , but x is 64, so we need to perform conv on x so that aligning the number of channel
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        x = self.doubleConv(x)

        if self.downsample:
            shortcut = self.downsample(shortcut)

        x += shortcut
        return self.relu(x)


# ResNet34 as encoder , UNet as decoder
class ResNet34_UNet(nn.Module):

    def __init__(
        self,
        input_channel=3,
        output_channel=1,
        channels=[64, 128, 256, 512],
        layers=[3, 4, 6, 3],
    ):

        super(ResNet34_UNet, self).__init__()
        # -----------------ResNet34 Building Begin-----------------
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, channels[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        self.layer1 = self._make_layer(channels[0], channels[0], layers[0], stride=1)
        self.layer2 = self._make_layer(channels[0], channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], layers[3], stride=2)

        # -----------------ResNet34 Building End-----------------

        # ----------------Unet expansive phase Begin----------

        self.bottom = DoubleConv(channels[-1], channels[-2])  # Channel : 512 -> 256

        self.expansive_blocks = nn.ModuleList()

        self.expansive_blocks.append(Unet_Expansive_Block(256, 512, 32))
        # On paper , it write 32 + 512 , but it should be 32 + 256(output of resnet34 layer 3)
        self.expansive_blocks.append(Unet_Expansive_Block(32, 256, 32))
        self.expansive_blocks.append(Unet_Expansive_Block(32, 128, 32))
        self.expansive_blocks.append(Unet_Expansive_Block(32, 64, 32))

        # Perform one more up-convolution to restore the original spatial size (e.g., 256x256)

        self.output = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),  # 32 -> 64
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channel, kernel_size=1),
            nn.Sigmoid(),
        )
        # ----------------Unet expansive phase End----------

    def _make_layer(self, in_channel, out_channel, num_layer, stride=1):

        downsample = None

        
        if stride != 1:
            # Increasing the number of channel and halve the size of input data
            downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel),
            )
            
        layers = []
        layers.append(Residual_Block(in_channel, out_channel, stride, downsample))
        for i in range(1, num_layer):
            layers.append(Residual_Block(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        skips = []
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        skips.append(x)
        x = self.layer2(x)
        skips.append(x)
        x = self.layer3(x)
        skips.append(x)
        x = self.layer4(x)
        skips.append(x)

        x = self.bottom(x)

        for i, block in enumerate(self.expansive_blocks):
            x = block(x, skips[-(i + 1)])

        x = self.output(x)

        return x
