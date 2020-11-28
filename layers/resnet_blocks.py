from typing import Optional
import torch.nn as nn

from layers.wrappers import conv1x1, conv3x3


def shortcut(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.AvgPool2d(stride=2, kernel_size=2), conv1x1(in_channels, out_channels)
    )


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

        self.block = nn.Sequential(
            conv1x1(in_channels, bn_channels),
            nn.BatchNorm2d(bn_channels),
            nn.ReLU(inplace=True),
            conv3x3(bn_channels, bn_channels, stride),
            nn.BatchNorm2d(bn_channels),
            nn.ReLU(inplace=True),
            conv1x1(bn_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )

        if self.downsample():
            self.shortcut = shortcut(in_channels, out_channels)
        else:
            self.shortcut = None

    def downsample(self):
        return self.in_channels != self.out_channels

    def forward(self, x):
        out = self.block(x)

        if self.shortcut is None:
            self.shortcut = x

        out += self.shortcut
        out = F.relu(out)
        return out