"""
This module contains an interface to accomidate both ResNet
and XResNet models.
"""

from typing import Optional, List, Type, Callable
from abc import ABCMeta

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.resnet_blocks import BottleneckBlock
from layers.xresnet_blocks import FastStem, tricked_bottleneck_block
from utils.weight_init import init_bn, init_c2msr_fill, init_cnn


class ResNetInterface(nn.Module, metaclass=ABCMeta):
    """
    Interface for subclassing all variants of ResNet and XResNet
    while supporting FPN use-case.
    """

    def __init__(
        self,
        layers: List[int],
        stem: nn.Module,
        block_func: Callable[..., BottleneckBlock],
        out_features: Optional[List[str]] = None,
        num_classes: Optional[int] = None,
        train_mode=False,
        use_dropout=False,
    ):
        super().__init__()
        self.inplanes = 64
        self.train_mode = train_mode
        self.use_dropout = use_dropout
        self.num_classes = num_classes
        self.stem = stem

        self._make_all_stages(layers, block_func)

        if self._do_classification():
            self._create_fc_layer()
            name = "fc"

            if out_features is None:
                out_features = [name]

        self._out_features = out_features

        init_cnn(self)
        init_bn(self)

    def _make_all_stages(self, layers, block_func):
        self.stage_names, self.stages = [], []
        for i, num_blocks in enumerate(layers):
            stride = 2
            self.num_channels = 2 ** i * 256
            name = "res" + str(i + 2)
            if name == "res2":
                stride = 1

            stage = self._make_layer(
                block_func, self.num_channels, num_blocks, stride=stride
            )

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

    def _make_layer(
        self,
        block_func: Callable[..., BottleneckBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
    ):
        layer = []
        for _ in range(blocks):
            layer.append(block_func(self.inplanes, planes, planes // 4, stride=stride))
            stride = 1
            self.inplanes = planes

        return nn.Sequential(*layer)

    def _do_classification(self):
        return self.num_classes is not None

    def _create_fc_layer(self):
        self.global_avg_pooling = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(self.num_channels, self.num_classes)

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
            if self.use_dropout:
                out = F.dropout(out)
            out = self.fc(out)

            if self.train_mode:
                return out

            if "fc" in self._out_features:
                outputs["fc"] = out

        return outputs
