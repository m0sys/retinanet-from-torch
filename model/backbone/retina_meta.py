import math
import torch.nn.functional as F
import torch.nn as nn

from layers.upsample import LateralUpsampleMerge
from layers.wrappers import conv3x3, conv1x1
from utils.weight_init import c2_xavier_fill


class RetinaNetFPN50(nn.Module):
    """
    Implements FPN network assuming a ResNet50 backbone.
    """

    def __init__(self, out_features=256):
        super().__init__()

        # Stage 7:
        self.conv7_up = conv3x3(out_features, out_features, stride=2)

        # Stage 6:
        self.conv6_up = conv3x3(512, out_features, stride=2)

        # Stage 5:
        self.lateral5 = conv1x1(512, out_features)
        self.conv5 = conv3x3(out_features, out_features)

        # Stage 4:
        self.lat_merge4 = LateralUpsampleMerge(256, out_features)
        self.conv4 = conv3x3(out_features, out_features)

        # Stage 3:
        self.lat_merge3 = LateralUpsampleMerge(128, out_features)
        self.conv3 = conv3x3(out_features, out_features)

        self.init_weights()

    def init_weights(self):
        """
        Initalize Conv layers with a gain of 1.

        The gain of 1 is due to us not applying any activation function.
        So we simply treat the conv layers as linear.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                c2_xavier_fill(m)

    def forward(self, C3, C4, C5):

        # Stage 6 and 7 forward.
        P6 = self.conv6_up(C5)
        P7 = self.conv7_up(F.relu(P6))

        # Stage 5 forward.
        out = self.lateral5(C5)
        P5 = self.conv5(out)

        # Stage 4 forward.
        out = self.lat_merge4(out, C4)
        P4 = self.conv4(out)

        # Stage 3 forward.
        out = self.lat_merge3(out, C3)
        P3 = self.conv3(out)

        return P3, P4, P5, P6, P7


class RetinaNetHead(nn.Module):
    """
    Implements RetinaNet head. see: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, num_classes, num_anchors=9, num_channels=256, prior=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_channels = num_channels

        self.classifier_subnet = nn.Sequential(
            conv3x3(self.num_channels, self.num_channels),
            nn.ReLU(inplace=True),
            conv3x3(self.num_channels, self.num_channels),
            nn.ReLU(inplace=True),
            conv3x3(self.num_channels, self.num_channels),
            nn.ReLU(inplace=True),
            conv3x3(self.num_channels, self.num_channels),
            nn.ReLU(inplace=True),
            conv3x3(self.num_channels, self.num_anchors * self.num_classes),
        )

        self.regressor_subnet = nn.Sequential(
            conv3x3(self.num_channels, self.num_channels),
            nn.ReLU(inplace=True),
            conv3x3(self.num_channels, self.num_channels),
            nn.ReLU(inplace=True),
            conv3x3(self.num_channels, self.num_channels),
            nn.ReLU(inplace=True),
            conv3x3(self.num_channels, self.num_channels),
            nn.ReLU(inplace=True),
            conv3x3(self.num_channels, self.num_anchors * 4),
        )

        # Initalize conv layers. see meth: `init_weights` for more details.
        self.init_weights()
        bias_value = -math.log((1 - prior) / prior)
        nn.init.constant_(self.classifier_subnet[-1].bias, bias_value)

    def init_weights(self):
        """
        Initialize weights as described in "Focal Loss for Dense
        Object Detection" Section 4.1 -> `Initialization`.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

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