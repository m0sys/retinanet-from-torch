from typing import Optional
import torch.nn as nn


def conv3x3(in_channels: int, out_channels: int, stride: Optional[int] = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)


def conv1x1(in_channels: int, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)


def half_max_pool2d():
    """
    Shrink 2d feature map's width and height by a factor of 2.
    """
    return nn.MaxPool2d(kernel_size=2, stride=2)