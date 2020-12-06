from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from layers.resnet_blocks import BottleNeckBlock, FastStem, StandardStem


class ResNet50(BaseModel):
    def __init__(self, num_classes: Optional[int] = None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = 2048

        self.stem = FastStem()

        if self._do_classification():
            self.global_avg_pooling = nn.AvgPool2d(kernel_size=7)
            self.fc = nn.Linear(self.num_features, self.num_classes)

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

    def _do_classification(self):
        return self.num_classes is not None

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
            return F.log_softmax(self.fc(out))

        return C2, C3, C4, C5  # output format for FPN
