from typing import Optional, Type, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from torchvision.models import resnet50

from model.backbone.resnet_interface import ResNetInterface
from layers.resnet_blocks import standard_bottleneck_block, StandardStem, BottleneckBlock


# The Good Implementation.

class ResNet(ResNetInterface):
    """
    Base class for creating all variants of ResNet while supporting FPN use-case.

    XResNet is derived from the following paper:
    "Bag of Tricks for Image Classification with Convolutional Neural Networks"

    """
    def __init__(self,
                 layers: List[int],
                 out_features: Optional[List[str]] = None,
                 num_classes: Optional[int] = None,
                 train_mode=False,
                 use_dropout=False,
        ):
        stem = StandardStem(use_dropout=use_dropout)
        super().__init__(layers, stem, standard_bottleneck_block, out_features, num_classes, train_mode, use_dropout)


def resnet50(out_features: Optional[List[str]] = None,
              num_classes: Optional[int] = None, train_mode=False, use_dropout=False):
    """Create a ResNet model 50 layers deep."""
    return ResNet(_RESNET50_LAYERS, out_features, num_classes, train_mode, use_dropout)


def resnet101(out_features: Optional[List[str]] = None,
               num_classes: Optional[int] = None, train_mode=False, use_dropout=False):
    """Create a ResNet model 101 layers deep."""
    return ResNet(_RESNET101_LAYERS, out_features, num_classes, train_mode, use_dropout)


def resnet152(out_features: Optional[List[str]] = None,
               num_classes: Optional[int] = None, train_mode=False, use_dropout=False):
    """Create a ResNet model 152 layers deep."""
    return ResNet(_RESNET152_LAYERS, out_features, num_classes, train_mode, use_dropout)


_RESNET34_LAYERS = [3, 4, 6, 3]
_RESNET50_LAYERS = [3, 4, 6, 3]
_RESNET101_LAYERS = [3, 4, 23, 3]
_RESNET152_LAYERS = [3, 8, 36, 3]


# Everything to get rid of.

class _ResNet(BaseModel):
    "Base ResNet module for all ResNets to inherit from."
    def __init__(self, stem: nn.Module, block: Type[BottleneckBlock], layers: List[int], out_features: List[str],  num_classes: Optional[int] = None):
        """
        Args:
            stem: a stem module - usually `FastStem` or `StandardStem`.
            block: a class denoting which type of block to use - a bottleneck block or a
                basic block.
            layers: a list representing the number of  blocks for each layer.
            out_features: a list of layer names whose outputs should be returned in `forward`.
            num_classes: if None will not perform classificaiton.
        """
        super().__init__()

        self.stem = stem
        self.block = block
        self.layers = layers
        self.out_features = out_features
        self.num_classes = num_classes

        if _do_classification():
            _create_fc_layer(self, num_classes)


    def _do_classification(self):
        return self.num_classes is not None

    def _create_fc_layer(self, num_classes):
        self.global_avg_pooling = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(self.num_features, num_classes)

        # Sec5.1 in "Accurate, Large Minibatch SGD: Training ImageNet
        # in 1 Hour."
        nn.init.normal_(self.fc.weight, std=0.01)


class ResNet50(BaseModel):
    """Standard ResNet50 model."""

    def __init__(self, pretrained=False, num_classes: Optional[int] = None):
        super().__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes

        self.num_features = 2048

        self.stem = StandardStem()

        if self._do_classification():
            self._create_fc_layer(self.num_classes)

        # Stage 2:
        self.layer2 = nn.Sequential(
            standard_bottleneck_block(64, 256, 64),
            standard_bottleneck_block(256, 256, 64),
            standard_bottleneck_block(256, 256, 64),
        )

        # Stage 3:
        self.layer3 = nn.Sequential(
            standard_bottleneck_block(256, 512, 128, stride=2),
            standard_bottleneck_block(512, 512, 128),
            standard_bottleneck_block(512, 512, 128),
            standard_bottleneck_block(512, 512, 128),
        )

        # Stage 4:
        self.layer4 = nn.Sequential(
            standard_bottleneck_block(512, 1024, 256, stride=2),
            standard_bottleneck_block(1024, 1024, 256),
            standard_bottleneck_block(1024, 1024, 256),
            standard_bottleneck_block(1024, 1024, 256),
            standard_bottleneck_block(1024, 1024, 256),
            standard_bottleneck_block(1024, 1024, 256),
        )

        # Stage 5:
        self.layer5 = nn.Sequential(
            standard_bottleneck_block(1024, 2048, 512, stride=2),
            standard_bottleneck_block(2048, 2048, 512),
            standard_bottleneck_block(2048, 2048, 512),
        )


        ## if self.pretrained:
        ##     self.preload_model()

    def init_weights(self):
        # PyTorch ResNet init.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _do_classification(self):
        return self.num_classes is not None

    def _create_fc_layer(self, num_classes):
        self.global_avg_pooling = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(self.num_features, num_classes)

        # Sec5.1 in "Accurate, Large Minibatch SGD: Training ImageNet
        # in 1 Hour."
        nn.init.normal_(self.fc.weight, std=0.01)

    def preload_model(self):
        self._create_fc_layer(1000)
        pretrained_model = resnet50(pretrained=True)
        pretrained_stats = list(pretrained_model.state_dict().items())
        stats_dict = self.state_dict()

        count = 0
        for k, _ in stats_dict.items():
            _, weights = pretrained_stats[count]
            stats_dict[k] = weights
            count += 1

    def forward(self, x):
        # Stage 1 forward.
        out = self.stem(x)

        # Stage 2 forwards.
        out = self.layer2(out)
        C2 = out

        # Stage 3 forwards
        out = self.layer3(out)
        C3 = out

        # Stage 4 forward.
        out = self.layer4(out)
        C4 = out

        # Stage 5 forward.
        out = self.layer5(out)
        C5 = out

        if self._do_classification():
            out = self.global_avg_pooling(out)
            out = torch.flatten(out, start_dim=1)
            return self.fc(out)

        return C2, C3, C4, C5  # output format for FPN


class XResNet50(BaseModel):
    """
    Bag of Tricks ResNet model

    This model is derived from the following paper:
    "Bag of Tricks for Image Classification with Convolutional Neural Networks"
    """
