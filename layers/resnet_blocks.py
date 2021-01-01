"""
This module contains block-level implementation for the standard ResNet model.

For more details see `paper`: "Deep Residual Learning for Image Recognition" @ https://arxiv.org/abs/1512.03385
"""

from typing import Callable
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


class BottleneckBlock(nn.Module):
    """Bottleneck Block used with resnet 50+."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        shortcut_func: Callable[..., nn.Sequential],
        stride: int = 1,
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
