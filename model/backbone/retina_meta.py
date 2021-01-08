import pdb
import math
import torch.nn as nn

from layers.wrappers import conv3x3


class RetinaNetHead(nn.Module):
    """
    Implements RetinaNet head. see: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, num_classes, num_anchors=9, num_channels=256, prior=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_channels = num_channels
        self.relu = nn.ReLU(inplace=True)

        self.classifier_subnet = self._create_subnet(self.num_classes)
        self.regressor_subnet = self._create_subnet(4)

        self.init_weights()

        # Apply the prior to bias to improve early training stability.
        ## pdb.set_trace()
        bias_value = -math.log((1 - prior) / prior)
        nn.init.constant_(self.classifier_subnet[-1].bias, bias_value)
        ## nn.init.constant_(self.regressor_subnet[-1].bias, bias_value)

    def _create_subnet(self, out: int):
        return nn.Sequential(
            conv3x3(self.num_channels, self.num_channels),
            self.relu,
            conv3x3(self.num_channels, self.num_channels),
            self.relu,
            conv3x3(self.num_channels, self.num_channels),
            self.relu,
            conv3x3(self.num_channels, self.num_channels),
            self.relu,
            conv3x3(self.num_channels, self.num_anchors * out),
        )

    def init_weights(self):
        """
        Initialize weights as described in "Focal Loss for Dense
        Object Detection" Section 4.1 -> `Initialization`.
        """

        for m in self.modules():
            self._init_focal_subnet(m)

    @classmethod
    def _init_focal_subnet(cls, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0.0)
        for l in m.children():
            cls._init_focal_subnet(l)

    def forward(self, P3, P4, P5, P6, P7):
        # Notice that each cell in the feature maps will output `num_anchors` * 4
        # dim channels as bbox preds & `num_anchors` * `num_classes` dim channels
        # as class preds.

        logits = {
            "p3": self.classifier_subnet(P3),
            "p4": self.classifier_subnet(P4),
            "p5": self.classifier_subnet(P5),
            "p6": self.classifier_subnet(P6),
            "p7": self.classifier_subnet(P7),
        }
        bbox_reg = {
            "p3": self.regressor_subnet(P3),
            "p4": self.regressor_subnet(P4),
            "p5": self.regressor_subnet(P5),
            "p6": self.regressor_subnet(P6),
            "p7": self.regressor_subnet(P7),
        }

        return logits, bbox_reg