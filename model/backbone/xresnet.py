"""
This module contains implementation for the XResNet model.

For more details see `paper`: "Bag of Tricks for Image Classification
with Convolutional Neural Networks" @ https://arxiv.org/abs/1812.01187
"""

from typing import Optional, List

from model.backbone.resnet_interface import ResNetInterface
from layers.xresnet_blocks import FastStem, tricked_bottleneck_block


class XResNet(ResNetInterface):
    """
    Base class for creating all variants of XResNet while supporting FPN use-case.

    XResNet is derived from the following paper:
    "Bag of Tricks for Image Classification with Convolutional Neural Networks"

    """

    def __init__(
        self,
        layers: List[int],
        out_features: Optional[List[str]] = None,
        num_classes: Optional[int] = None,
        train_mode=False,
        use_dropout=False,
    ):
        stem = FastStem(use_dropout=use_dropout)
        super().__init__(
            layers,
            stem,
            tricked_bottleneck_block,
            out_features,
            num_classes,
            train_mode,
            use_dropout,
        )


def xresnet50(
    out_features: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
    train_mode=False,
    use_dropout=False,
):
    """Create a XResNet model 50 layers deep."""
    return XResNet(_RESNET50_LAYERS, out_features, num_classes, train_mode, use_dropout)


def xresnet101(
    out_features: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
    train_mode=False,
    use_dropout=False,
):
    """Create a XResNet model 101 layers deep."""
    return XResNet(
        _RESNET101_LAYERS, out_features, num_classes, train_mode, use_dropout
    )


def xresnet152(
    out_features: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
    train_mode=False,
    use_dropout=False,
):
    """Create a XResNet model 152 layers deep."""
    return XResNet(
        _RESNET152_LAYERS, out_features, num_classes, train_mode, use_dropout
    )


_RESNET34_LAYERS = [3, 4, 6, 3]
_RESNET50_LAYERS = [3, 4, 6, 3]
_RESNET101_LAYERS = [3, 4, 23, 3]
_RESNET152_LAYERS = [3, 8, 36, 3]
