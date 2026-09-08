from typing import TypedDict

from torch import cat, nn


class DoubleConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,  
        ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class DownSample(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
        ):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x):
        x = self.max_pool(x)
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
        ):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(
            kernel_size=2,
            in_channels=in_channels,
            out_channels=in_channels // 2,
            stride=2,
        )
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x, x_down):
        x_up = self.up_conv(x)
        x = cat((x_down, x_up), 1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.double_convolution = DoubleConv(in_channels, 64)
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)
        self.down4 = DownSample(512, 1024)
        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        dc = self.double_convolution(x)
        down_1 = self.down1(dc)
        down_2 = self.down2(down_1)
        down_3 = self.down3(down_2)
        down_4 = self.down4(down_3)
        up_1 = self.up1(down_4, down_3)
        up_2 = self.up2(up_1, down_2)
        up_3 = self.up3(up_2, down_1)
        up_4 = self.up4(up_3, dc)
        out = self.final_conv(up_4)

        return out

class UNetParams(TypedDict):
    in_channels: int
    out_channels: int
