"""
This module contains block-level implementation for the XResNet model.

For more details see `paper`: "Bag of Tricks for Image Classification
with Convolutional Neural Networks" @ https://arxiv.org/abs/1812.01187
"""

import torch.nn as nn
from layers.wrappers import conv1x1, conv3x3, half_max_pool2d
from layers.resnet_blocks import BottleneckBlock


class FastStem(nn.Module):
    """
    Resnet low-level features optimized by using 3x3 convs.

    This stem is derived from  "Bag of Tricks for Image
    Classification with Convolutional Neural Networks" paper.

    Dropout Notes:
        - When `use_dropout` is true apply dropout to each layer.
        This is done in accordance to the original Dropout paper
        which states that dropout in the lower layers still helps by
        providing noisy inputs for the higher fully connected layers.
        For more details see:
            https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf

        - This can help prevent overfitting if training is done on a small
        dataset.
    """

    def __init__(self, in_channels=3, out_channels=64, use_dropout=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_dropout = use_dropout

        self.conv1_1 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.conv1_2 = conv3x3(out_channels // 2, out_channels // 2, use_bias=False)
        self.conv1_3 = conv3x3(out_channels // 2, out_channels, use_bias=False)
        self.bn1_1 = nn.BatchNorm2d(out_channels // 2)
        self.bn1_2 = nn.BatchNorm2d(out_channels // 2)
        self.bn1_3 = nn.BatchNorm2d(out_channels)
        self.pool1 = half_max_pool2d()
        self.relu = nn.LeakyReLU(inplace=True)

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
        out = self.relu(self.bn1_1(self.conv1_1(x)))
        out = self._apply_dropout(out)

        out = self.relu(self.bn1_2(self.conv1_2(out)))
        out = self._apply_dropout(out)

        out = self.relu(self.bn1_3(self.conv1_3(out)))
        out = self._apply_dropout(out)

        return self.pool1(out)


def tricked_bottleneck_block(
    in_channels: int, out_channels: int, bn_channels, stride=1
):
    """Return `BottleneckBlock` with tricked shortcut func."""
    return BottleneckBlock(
        in_channels, out_channels, bn_channels, tricked_shortcut, stride=stride
    )


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
