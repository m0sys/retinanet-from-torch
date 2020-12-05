from typing import Optional, Type
import torch.nn as nn

from base import BaseModel
from model.backbone.resnet import ResNet50
from model.backbone.retina_meta import RetinaNetFPN50, RetinaNetHead
from model.anchor_generator import AnchorBoxGenerator
from utils.shape_utils import permute_to_N_HWA_K


class RetinaNet500(BaseModel):
    def __init__(self, num_classes: Optional[int] = 80):
        super().__init__()

        anchor_gen = AnchorBoxGenerator(
            sizes=[32.0, 64.0, 128.0, 256.0, 512.0],
            aspect_ratios=[0.5, 1.0, 2.0],
            scales=[1.0, 2 ** (1 / 3), 2 ** (2 / 3)],
            strides=[2, 2, 2, 2, 2],
        )

        self.model = _RetinaNet(
            ResNet50, RetinaNetFPN50, RetinaNetHead, anchor_gen, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


class _RetinaNet(nn.Module):
    def __init__(
        self,
        base: Type[nn.Module],
        backbone: Type[nn.Module],
        head: Type[nn.Module],
        anchor_generator: AnchorBoxGenerator,
        num_classes=20,
    ):
        super().__init__()
        self.base = base()
        self.backbone = backbone()
        self.head = head(num_classes)
        self.anchor_generator = anchor_generator
        self.num_classes = num_classes

    def forward(self, x):
        _, C3, C4, C5 = self.base(x)
        P3, P4, P5, P6, P7 = self.backbone(C3, C4, C5)

        pred_logits, pred_bboxes = self.head(P3, P4, P5, P6, P7)

        anchors = self.anchor_generator([P3, P4, P5, P6, P7])

        reshaped_logits = [
            permute_to_N_HWA_K(pred_logits[k], self.num_classes) for k in pred_logits
        ]

        reshaped_bboxes = [permute_to_N_HWA_K(pred_bboxes[k], 4) for k in pred_bboxes]

        return reshaped_logits, reshaped_bboxes, anchors


## class RetinaNet(BaseModel):
##     def __init__(self, num_classes: Optional[int] = 80):
##         super().__init__()
##
##         sizes = [32.0, 64.0, 128.0, 256.0, 512.0]
##         aspect_ratios = [0.5, 1.0, 2.0]
##         scales = [1.0, 2 ** (1 / 3), 2 ** (2 / 3)]
##         strides = [2, 2, 2, 2, 2]
##
##         self.base = ResNet50()
##         self.backbone = RetinaNetFPN50()
##         self.head = RetinaNetHead(num_classes)
##         self.anchors = AnchorBoxGenerator(sizes, aspect_ratios, strides, scales)
##
##     def forward(self, x):
##         _, C3, C4, C5 = self.base(x)
##         P3, P4, P5, P6, P7 = self.backbone(C3, C4, C5)
##
##         return self.head(P3, P4, P5, P6, P7)