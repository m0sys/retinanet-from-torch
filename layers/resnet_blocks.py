from typing import Optional, Callable
import torch.nn as nn
import torch.nn.functional as F

from layers.wrappers import conv1x1, conv3x3, half_max_pool2d
from utils.weight_init import init_c2msr_fill


class StandardStem(nn.Module):
    """
    Standard 7x7 Conv stem followed by MaxPool.
    """

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=7, stride=2,
            padding=3, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = half_max_pool2d()

        init_c2msr_fill(self)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv1(x))))


class FastStem(nn.Module):
    """
    Resnet low-level features optimized by using 3x3 convs.

    This stem is derived from  "Bag of Tricks for Image
    Classification with Convolutional Neural Networks" paper.
    """

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1_1 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.conv1_2 = conv3x3(out_channels // 2, out_channels // 2,
                               use_bias=False)
        self.conv1_3 = conv3x3(out_channels // 2, out_channels, use_bias=False)
        self.bn1_1 = nn.BatchNorm2d(out_channels // 2)
        self.bn1_2 = nn.BatchNorm2d(out_channels // 2)
        self.bn1_3 = nn.BatchNorm2d(out_channels)
        self.pool1 = half_max_pool2d()

        init_c2msr_fill(self)

    def forward(self, x):
        out = F.relu(self.bn1_1(self.conv1_1(x)))
        out = F.relu(self.bn1_2(self.conv1_2(out)))
        out = F.relu(self.bn1_3(self.conv1_3(out)))

        return self.pool1(out)


class BottleneckBlock(nn.Module):
    """Bottleneck Block used with resnet 50+."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        shortcut_func: Callable[..., nn.Sequential],
        stride: int = 1
    ):
        """
        Args:
            bn_channels (int): number of output channels for the 3x3.
            shortcut_func: a callable function that returns a `nn.Sequential`
                which describes the kind of shortcut to use.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn_channels = bn_channels
        self.stride = stride

        self.block = nn.Sequential(
            conv1x1(in_channels, bn_channels, use_bias=False),
            nn.BatchNorm2d(bn_channels),
            nn.ReLU(inplace=True),
            conv3x3(bn_channels, bn_channels, stride, use_bias=False),
            nn.BatchNorm2d(bn_channels),
            nn.ReLU(inplace=True),
            conv1x1(bn_channels, out_channels, use_bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if self.downsample():
            self.shortcut = shortcut_func(in_channels, out_channels, stride=stride)
        else:
            self.shortcut = None

        init_c2msr_fill(self)

    def downsample(self):
        return self.in_channels != self.out_channels

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = F.relu(out)
        return out


def tricked_bottleneck_block(in_channels: int, out_channels: int,
                             bn_channels, stride=1):
    """Return `BottleneckBlock` with tricked shortcut func."""
    return BottleneckBlock(in_channels, out_channels,
                           bn_channels, tricked_shortcut, stride=stride)


def tricked_shortcut(in_channels: int, out_channels: int, stride: int = 1):
    """
    ResNet shortcut connection (path b) as described in
    "Bag of Tricks for Image Classification with Convolutional
    Neural Networks" paper.

    For more details see: https://arxiv.org/abs/1812.01187
    """

    if stride != 1:
        return nn.Sequential(
            nn.AvgPool2d(stride=stride, kernel_size=2),
            conv1x1(in_channels, out_channels, use_bias=False),
            nn.BatchNorm2d(out_channels),
        )

    else:
        return nn.Sequential(
            conv1x1(in_channels, out_channels, use_bias=False),
            nn.BatchNorm2d(out_channels),
        )


def standard_bottleneck_block(in_channels: int,
                              out_channels: int,
                              bn_channels, stride=1):
    """Return a `BottleneckBlock` with standard shortcut func."""
    return BottleneckBlock(in_channels, out_channels,
                           bn_channels, standard_shortcut, stride=stride)


def standard_shortcut(in_channels: int, out_channels: int, stride: int = 1):
    """
    Standard ResNet shortcut connection.
    """

    return nn.Sequential(
        conv1x1(
            in_channels,
            out_channels,
            stride=stride,
            use_bias=False
        ),
        nn.BatchNorm2d(out_channels),
    )
