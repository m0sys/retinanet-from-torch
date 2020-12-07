from typing import Optional
import torch.nn as nn
import torch.nn.functional as F

from layers.wrappers import conv1x1, conv3x3, half_max_pool2d
from utils.weight_init import c2_msra_fill


def init_cnn(m):
    """
    FastAI XResNet Init.

    see: https://github.com/fastai/fastai/blob/cad02a84e1adfd814bd97ff833a0d9661516308d/fastai/vision/models/xresnet.py#L16
    """
    if getattr(m, "bias", None) is not None:
        nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
    for l in m.children():
        init_cnn(l)


def tricked_shortcut(in_channels: int, out_channels: int):
    """
    ResNet shortcut connection (path b) as described in
    "Bag of Tricks for Image Classification with Convolutional
    Neural Networks" paper.

    For more details see: https://arxiv.org/abs/1812.01187
    """
    return nn.Sequential(
        nn.AvgPool2d(stride=2, kernel_size=2),
        conv1x1(in_channels, out_channels, use_bias=False),
        nn.BatchNorm2d(out_channels),
    )


def nondownsample_shortcut(in_channels: int, out_channels: int):
    return nn.Sequential(
        conv1x1(in_channels, out_channels, use_bias=False), nn.BatchNorm2d(out_channels)
    )


class StandardStem(nn.Module):
    """
    Standard 7x7 Resnet stem followed by pooling.
    """

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = half_max_pool2d()

        ## c2_msra_fill(self.conv1)

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
        self.conv1_2 = conv3x3(out_channels // 2, out_channels // 2, use_bias=False)
        self.conv1_3 = conv3x3(out_channels // 2, out_channels, use_bias=False)
        self.bn1_1 = nn.BatchNorm2d(out_channels // 2)
        self.bn1_2 = nn.BatchNorm2d(out_channels // 2)
        self.bn1_3 = nn.BatchNorm2d(out_channels)
        self.pool1 = half_max_pool2d()

        ## self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

    def forward(self, x):
        out = F.relu(self.bn1_1(self.conv1_1(x)))
        out = F.relu(self.bn1_2(self.conv1_2(out)))
        out = F.relu(self.bn1_3(self.conv1_3(out)))

        return self.pool1(out)


class BottleNeckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        stride: Optional[int] = 1,
    ):
        """
        Args:
            bn_channels (int): number of output channels for the 3x3.
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
            self.shortcut = tricked_shortcut(in_channels, out_channels)
        elif self.scale_identity():
            self.shortcut = nondownsample_shortcut(in_channels, out_channels)
        else:
            self.shortcut = None

        ## self.init_weights()

    def downsample(self):
        ## return self.in_channels != self.out_channels
        return self.stride != 1

    def scale_identity(self):
        return self.in_channels != self.out_channels

    def init_weights(self):
        """Initializes Conv layer weights using He Init with `fan_out`."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)  # detectron init

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = F.relu(out)
        return out