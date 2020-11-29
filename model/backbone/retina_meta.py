import torch.nn.functional as F
import torch.nn as nn

from layers.upsample import LateralUpsampleMerge
from layers.wrappers import conv3x3, conv1x1


class RetinaNetFPN50(nn.Module):
    """
    Implements FPN network assuming a ResNet50 backbone.
    """

    def __init__(self, out_features=256):
        super().__init__()

        # Stage 7:
        self.conv7_up = conv3x3(out_features, out_features, stride=2)

        # Stage 6:
        self.conv6_up = conv3x3(2048, out_features, stride=2)

        # Stage 5:
        self.lateral5 = conv1x1(2048, out_features)
        self.conv5 = conv3x3(out_features, out_features)

        # Stage 4:
        self.lat_merge4 = LateralUpsampleMerge(1024, out_features)
        self.conv4 = conv3x3(out_features, out_features)

        # Stage 3:
        self.lat_merge3 = LateralUpsampleMerge(512, out_features)
        self.conv3 = conv3x3(out_features, out_features)

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

    def __init__(self, num_classes, num_anchors=9, num_channels=256):
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

    def forward(self, P3, P4, P5, P6, P7):

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