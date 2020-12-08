"""Implementation of XResNet."""
from typing import Optional, List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.resnet_blocks import BottleneckBlock, FastStem, init_c2msr_fill

class XResNet(nn.Module):
    def __init__(self,
                 layers: List[int],
                 out_features: Optional[List[str]] = None,
                 num_classes: Optional[int] = None
        ):
        super().__init__()
        self.inplanes = 64
        ## self.num_features = 2 ** (len(layers) - 1) * 256

        self.num_classes = num_classes

        self.stem = FastStem()

        self.stage_names, self.stages = [], []
        for i, num_blocks in enumerate(layers):
            stride = 2
            self.num_channels = 2 ** i * 256
            name = "res" + str(i + 2)
            if name == "res2":
                stride = 1

            stage = self._make_layer(BottleneckBlock,
                                     self.num_channels,
                                     layers[i],
                                     stride=stride)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)


        ## self.layer1 = self._make_layer(BottleneckBlock, 256, layers[0])
        ## self.layer2 = self._make_layer(BottleneckBlock, 512, layers[1], stride=2)
        ## self.layer3 = self._make_layer(BottleneckBlock, 1024, layers[2], stride=2)
        ## self.layer4 = self._make_layer(BottleneckBlock, 2048, layers[3], stride=2)

        init_c2msr_fill(self)

        if self._do_classification():
            self._create_fc_layer(num_classes)
            name = "fc"

        if out_features is None:
            out_features = [name]

        self._out_features = out_features


    def _make_layer(self,
                    block: Type[BottleneckBlock],
                    planes: int, blocks: int,
                    stride: int = 1
                    ):
        layer = []
        for i in range(blocks):
            layer.append(BottleneckBlock(self.inplanes, planes, planes // 4, stride=stride))
            stride = 1
            self.inplanes = planes

        return nn.Sequential(*layer)

    def _do_classification(self):
        return self.num_classes is not None

    def _create_fc_layer(self, num_classes):
        self.global_avg_pooling = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(self.num_features, num_classes)

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
            out = torch.squeeze(out)
            if "fc" in self._out_features:
                outputs["fc"] = out

        return outputs

        ## # Stage 1 forward.
        ## out = self.stem(x)

        ## # Stage 2 forward.
        ## out = self.layer1(out)
        ## C2 = out

        ## # Stage 3 forwards
        ## out = self.layer2(out)
        ## C3 = out

        ## # Stage 4 forward.
        ## out = self.layer3(out)
        ## C4 = out

        ## # Stage 5 forward.
        ## out = self.layer4(out)
        ## C5 = out

        ## if self._do_classification():
        ##     out = self.global_avg_pooling(out)
        ##     out = torch.squeeze(out)
        ##     return self.fc(out)

        ## return C2, C3, C4, C5  # output format for FPN


RESNET34_LAYERS = [3, 4, 6, 3]
RESNET50_LAYERS = [3, 4, 6, 3]
RESNET101_LAYERS = [3, 4, 23, 3]
RESNET152_LAYERS = [3, 8, 36, 3]
