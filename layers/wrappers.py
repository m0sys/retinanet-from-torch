from typing import Optional

import torch
import torch.nn as nn


def conv3x3(
    in_channels: int, out_channels: int, stride: Optional[int] = 1, use_bias=True
):
    """
    Canonical 3 by 3 "Same" convolutional layer.
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=use_bias,
    )


def conv1x1(in_channels: int, out_channels, stride=1, use_bias=True):
    """
    Canonical 1 by 1 "Bottleneck" convolutional layer.
    """
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=use_bias
    )


def half_max_pool2d():
    """
    Shrink 2d feature map's width and height by a factor of 2.
    """
    return nn.MaxPool2d(kernel_size=2, stride=2)


def nonzero_tuple(x):
    """
    A 'as_tuple=True' version of torch.nonzero to support torchscript.
    because of https://github.com/pytorch/pytorch/issues/38718
    """
    if torch.jit.is_scripting():
        if x.dim() == 0:
            return x.unsqueeze(0).nonzero().unbind(1)
        return x.nonzero().unbind(1)
    else:
        return x.nonzero(as_tuple=True)
