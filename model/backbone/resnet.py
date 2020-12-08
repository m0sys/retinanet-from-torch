from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from torchvision.models import resnet50

from layers.resnet_blocks import BottleNeckBlock, FastStem, StandardStem, init_cnn


class ResNet50(BaseModel):
    """Standard ResNet50 model."""

    def __init__(self, pretrained=False, num_classes: Optional[int] = None):
        super().__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes

        self.num_features = 2048

        self.stem = StandardStem() if self.pretrained else FastStem()

        if self._do_classification():
            self._create_fc_layer(self.num_classes)

        # Stage 2:
        self.layer2 = nn.Sequential(
            BottleNeckBlock(64, 256, 64),
            BottleNeckBlock(256, 256, 64),
            BottleNeckBlock(256, 256, 64),
        )

        # Stage 3:
        self.layer3 = nn.Sequential(
            BottleNeckBlock(256, 512, 128, stride=2),
            BottleNeckBlock(512, 512, 128),
            BottleNeckBlock(512, 512, 128),
            BottleNeckBlock(512, 512, 128),
        )

        # Stage 4:
        self.layer4 = nn.Sequential(
            BottleNeckBlock(512, 1024, 256, stride=2),
            BottleNeckBlock(1024, 1024, 256),
            BottleNeckBlock(1024, 1024, 256),
            BottleNeckBlock(1024, 1024, 256),
            BottleNeckBlock(1024, 1024, 256),
            BottleNeckBlock(1024, 1024, 256),
        )

        # Stage 5:
        self.layer5 = nn.Sequential(
            BottleNeckBlock(1024, 2048, 512, stride=2),
            BottleNeckBlock(2048, 2048, 512),
            BottleNeckBlock(2048, 2048, 512),
        )

        ## self.init_weights()
        init_cnn(self)

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
        ## nn.init.normal_(self.fc.weight, std=0.01)

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
            out = torch.squeeze(out)
            return self.fc(out)

        return C2, C3, C4, C5  # output format for FPN


class XResNet50(BaseModel):
    """
    Bag of Tricks ResNet model

    This model is derived from the following paper:
    "Bag of Tricks for Image Classification with Convolutional Neural Networks"
    """