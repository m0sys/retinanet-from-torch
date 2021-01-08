from typing import Callable
import torch.nn as nn

from model.backbone.resnet import resnet50, resnet101, resnet152
from model.backbone.retina_meta import RetinaNetHead
from model.backbone.fpn import retinanet_fpn_resnet
from model.obj_utils.anchor_generator import AnchorBoxGenerator
from utils.shape_utils import permute_to_N_HWA_K


class _RetinaNet(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        backbone: nn.Module,
        head: nn.Module,
        anchor_generator: AnchorBoxGenerator,
        num_classes=80,
    ):
        super().__init__()
        self.base = base
        self.backbone = backbone
        self.head = head
        self.anchor_generator = anchor_generator
        self.num_classes = num_classes

    def forward(self, x):
        base_outputs = self.base(x)
        C3 = base_outputs["res3"]
        C4 = base_outputs["res4"]
        C5 = base_outputs["res5"]

        backbone_outputs = self.backbone([C3, C4, C5])
        P3 = backbone_outputs["fpn0"]
        P4 = backbone_outputs["fpn1"]
        P5 = backbone_outputs["fpn2"]
        P6 = backbone_outputs["upsample_fpn3"]
        P7 = backbone_outputs["upsample_fpn4"]

        pred_logits, pred_bboxes = self.head(P3, P4, P5, P6, P7)

        anchors = self.anchor_generator([P3, P4, P5, P6, P7])

        reshaped_logits = [
            permute_to_N_HWA_K(pred_logits[k], self.num_classes) for k in pred_logits
        ]

        reshaped_bboxes = [permute_to_N_HWA_K(pred_bboxes[k], 4) for k in pred_bboxes]

        outputs = {
            "pred_logits": reshaped_logits,
            "pred_bboxes": reshaped_bboxes,
            "anchors": anchors,
        }

        return outputs


def retina_resnet50(num_classes):
    return _retina_resnet(resnet50, num_classes)


def retina_resnet101(num_classes):
    return _retina_resnet(resnet101, num_classes)


def retina_resnet152(num_classes):
    return _retina_resnet(resnet152, num_classes)


def _retina_resnet(resnet_func: Callable[..., nn.Module], num_classes):
    base = resnet_func(["res3", "res4", "res5"], pretrained=True)
    backbone = retinanet_fpn_resnet()
    head = RetinaNetHead(num_classes=num_classes)
    anchor_gen = AnchorBoxGenerator()
    return _RetinaNet(base, backbone, head, anchor_gen, num_classes)
