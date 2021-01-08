"""
This module contains block-level implementation for the standard ResNet model.

For more details see `paper`: "Deep Residual Learning for Image Recognition" @ https://arxiv.org/abs/1512.03385
"""

from typing import Callable
import torch.nn as nn

from layers.wrappers import conv1x1, conv3x3, half_max_pool2d


class StandardStem(nn.Module):
    """
    Standard 7x7 Conv stem followed by MaxPool.
    """

    def __init__(self, in_channels=3, out_channels=64, use_dropout=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = half_max_pool2d()

        if use_dropout:
            self.dropout = nn.Dropout2d()
        else:
            self.dropout = None

    def _apply_dropout(self, x):
        if self.use_dropout:
            return self.dropout(x)
        else:
            return x

    def forward(self, x):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self._apply_dropout(out)


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

        self.conv1 = conv1x1(in_channels, bn_channels, use_bias=False)
        self.bn1 = nn.BatchNorm2d(bn_channels)
        self.conv2 = conv3x3(bn_channels, bn_channels, stride, use_bias=False)
        self.bn2 = nn.BatchNorm2d(bn_channels)
        self.conv3 = conv1x1(bn_channels, out_channels, use_bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        if self.do_downsample():
            self.downsample = shortcut_func(in_channels, out_channels, stride=stride)
        else:
            self.downsample = None

    def do_downsample(self):
        return self.in_channels != self.out_channels

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


def standard_bottleneck_block(
    in_channels: int, out_channels: int, bn_channels, stride=1
):
    """Return a `BottleneckBlock` with standard shortcut func."""
    return BottleneckBlock(
        in_channels, out_channels, bn_channels, standard_shortcut, stride=stride
    )


def standard_shortcut(in_channels: int, out_channels: int, stride: int = 1):
    """
    Standard ResNet shortcut connection.
    """

    return nn.Sequential(
        conv1x1(in_channels, out_channels, stride=stride, use_bias=False),
        nn.BatchNorm2d(out_channels),
    )
