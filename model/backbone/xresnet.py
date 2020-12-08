"""Implementation of XResNet."""
from typing import Optional, List, Type, Callable

import torch
import torch.nn as nn

from layers.resnet_blocks import FastStem, BottleneckBlock, tricked_bottleneck_block


class XResNet(nn.Module):
    """
    Base class for creating all variants of XResNet while supporting FPN use-case.

    XResNet is derived from the following paper:
    "Bag of Tricks for Image Classification with Convolutional Neural Networks"

    """
    def __init__(self,
                 layers: List[int],
                 out_features: Optional[List[str]] = None,
                 num_classes: Optional[int] = None
        ):
        super().__init__()
        self.inplanes = 64

        self.num_classes = num_classes

        self.stem = FastStem()

        self.stage_names, self.stages = [], []
        for i, num_blocks in enumerate(layers):
            stride = 2
            self.num_channels = 2 ** i * 256
            name = "res" + str(i + 2)
            if name == "res2":
                stride = 1

            stage = self._make_layer(tricked_bottleneck_block,
                                     self.num_channels,
                                     layers[i],
                                     stride=stride)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

        if self._do_classification():
            self._create_fc_layer(num_classes)
            name = "fc"

        if out_features is None:
            out_features = [name]

        self._out_features = out_features

    def _make_layer(self,
                    block_func: Callable[..., BottleneckBlock],
                    planes: int, blocks: int,
                    stride: int = 1
                    ):
        layer = []
        for _ in range(blocks):
            layer.append(block_func(self.inplanes, planes, planes // 4, stride=stride))
            stride = 1
            self.inplanes = planes

        return nn.Sequential(*layer)

    def _do_classification(self):
        return self.num_classes is not None

    def _create_fc_layer(self, num_classes):
        self.global_avg_pooling = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(self.num_channels, num_classes)

        # Sec5.1 in "Accurate, Large Minibatch SGD: Training ImageNet
        # in 1 Hour."
        nn.init.normal_(self.fc.weight, std=0.01)

    def forward(self, x):
        outputs = {}

        out = self.stem(x)

        if "stem" in self._out_features:
            outputs["stem"] = out

        for name, stage in zip(self.stage_names, self.stages):
            out = stage(out)
            if name in self._out_features:
                outputs[name] = out

        if self._do_classification():
            out = self.global_avg_pooling(out)
            out = torch.flatten(out, start_dim=1)
            if "fc" in self._out_features:
                outputs["fc"] = out

        return outputs


def xresnet50(out_features: Optional[List[str]] = None,
              num_classes: Optional[int] = None):
    """Create a XResNet model 50 layers deep."""
    return XResNet(_RESNET50_LAYERS, out_features, num_classes)


def xresnet101(out_features: Optional[List[str]] = None,
               num_classes: Optional[int] = None):
    """Create a XResNet model 101 layers deep."""
    return XResNet(_RESNET101_LAYERS, out_features, num_classes)


def xresnet152(out_features: Optional[List[str]] = None,
               num_classes: Optional[int] = None):
    """Create a XResNet model 152 layers deep."""
    return XResNet(_RESNET152_LAYERS, out_features, num_classes)


_RESNET34_LAYERS = [3, 4, 6, 3]
_RESNET50_LAYERS = [3, 4, 6, 3]
_RESNET101_LAYERS = [3, 4, 23, 3]
_RESNET152_LAYERS = [3, 8, 36, 3]
