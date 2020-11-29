from typing import Optional
from base import BaseModel

from model.backbone.resnet import ResNet50
from model.backbone.retina_meta import RetinaNetFPN50, RetinaNetHead
from model.anchor_generator import AnchorBoxGenerator


class ResnetModel(BaseModel):
    def __init__(self, num_classes: Optional[int] = 80):
        super().__init__()

        sizes = [32.0, 64.0, 128.0, 256.0, 512.0]
        aspect_ratios = [0.5, 1.0, 2.0]
        scales = [1.0, 2 ** (1 / 3), 2 ** (2 / 3)]
        strides = [2, 2, 2, 2, 2]

        self.base = ResNet50()
        self.backbone = RetinaNetFPN50()
        self.head = RetinaNetHead(num_classes)
        self.anchors = AnchorBoxGenerator(sizes, aspect_ratios, strides, scales)

    def forward(self, x):
        _, C3, C4, C5 = self.base(x)
        P3, P4, P5, P6, P7 = self.backbone(C3, C4, C5)

        return self.head(P3, P4, P5, P6, P7)