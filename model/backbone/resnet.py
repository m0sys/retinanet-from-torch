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
        self.num_features = 512

        self.stem = FastStem()

        if self._do_classification():
            self.global_avg_pooling = nn.AvgPool2d(kernel_size=7)
            self.fc = nn.Linear(self.num_features, self.num_classes)

        # Stage 2:
        self.res_block2_1 = BottleNeckBlock(64, 64, 32)
        self.res_block2_2 = BottleNeckBlock(64, 64, 32)
        self.res_block2_3 = BottleNeckBlock(64, 64, 32)

        # Stage 3:
        self.res_block3_1 = BottleNeckBlock(64, 128, 64, stride=2)
        self.res_block3_2 = BottleNeckBlock(128, 128, 64)
        self.res_block3_3 = BottleNeckBlock(128, 128, 64)
        self.res_block3_4 = BottleNeckBlock(128, 128, 64)

        # Stage 4:
        self.res_block4_1 = BottleNeckBlock(128, 256, 128, stride=2)
        self.res_block4_2 = BottleNeckBlock(256, 256, 128)
        self.res_block4_3 = BottleNeckBlock(256, 256, 128)
        self.res_block4_4 = BottleNeckBlock(256, 256, 128)
        self.res_block4_5 = BottleNeckBlock(256, 256, 128)
        self.res_block4_6 = BottleNeckBlock(256, 256, 128)

        # Stage 5:
        self.res_block5_1 = BottleNeckBlock(256, 512, 256, stride=2)
        self.res_block5_2 = BottleNeckBlock(512, 512, 256)
        self.res_block5_3 = BottleNeckBlock(512, 512, 256)

    def _do_classification(self):
        return self.num_classes is not None

    def forward(self, x):
        # Stage 1 forward.
        out = self.stem(x)

        # Stage 2 forwards.
        out = self.res_block2_1(out)
        out = self.res_block2_2(out)
        out = self.res_block2_3(out)
        C2 = out

        # Stage 3 forwards
        out = self.res_block3_1(out)
        out = self.res_block3_2(out)
        out = self.res_block3_3(out)
        out = self.res_block3_4(out)
        C3 = out

        # Stage 4 forward.
        out = self.res_block4_1(out)
        out = self.res_block4_2(out)
        out = self.res_block4_3(out)
        out = self.res_block4_4(out)
        out = self.res_block4_5(out)
        out = self.res_block4_6(out)
        C4 = out

        # Stage 5 forward.
        out = self.res_block5_1(out)
        out = self.res_block5_2(out)
        out = self.res_block5_3(out)
        C5 = out

        if self._do_classification():
            out = self.global_avg_pooling(out)
            out = torch.squeeze(out)
            return F.log_softmax(self.fc(out))

        return C2, C3, C4, C5  # output format for FPN
