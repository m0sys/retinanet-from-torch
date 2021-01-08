import pytest
import torch

from model.backbone.resnet import resnet50
from model.backbone.retina_meta import RetinaNetHead
from model.backbone.fpn import retinanet_fpn_resnet
from utils.shape_utils import permute_to_N_HWA_K


BATCH_SIZE = 1


@pytest.fixture(scope="module")
def init_512x512_dummy_data():
    return torch.randn((BATCH_SIZE, 3, 512, 512))


def test_retina_head_output_shapes(init_512x512_dummy_data):
    data = init_512x512_dummy_data
    num_anchors = 9
    num_classes = 20

    base = resnet50(out_features=["res3", "res4", "res5"])
    backbone = retinanet_fpn_resnet()
    head = RetinaNetHead(20)

    outputs = base(data)
    C3 = outputs["res3"]
    C4 = outputs["res4"]
    C5 = outputs["res5"]

    b_outs = backbone([C3, C4, C5])
    P3 = b_outs["fpn0"]
    P4 = b_outs["fpn1"]
    P5 = b_outs["fpn2"]
    P6 = b_outs["upsample_fpn3"]
    P7 = b_outs["upsample_fpn4"]

    pred_logits, pred_bboxes = head(P3, P4, P5, P6, P7)

    assert pred_logits["p3"].shape == (BATCH_SIZE, num_classes * num_anchors, 64, 64)
    assert pred_logits["p4"].shape == (BATCH_SIZE, num_classes * num_anchors, 32, 32)
    assert pred_logits["p5"].shape == (BATCH_SIZE, num_classes * num_anchors, 16, 16)
    assert pred_logits["p6"].shape == (BATCH_SIZE, num_classes * num_anchors, 8, 8)
    assert pred_logits["p7"].shape == (BATCH_SIZE, num_classes * num_anchors, 4, 4)

    assert pred_bboxes["p3"].shape == (BATCH_SIZE, 4 * num_anchors, 64, 64)
    assert pred_bboxes["p4"].shape == (BATCH_SIZE, 4 * num_anchors, 32, 32)
    assert pred_bboxes["p5"].shape == (BATCH_SIZE, 4 * num_anchors, 16, 16)
    assert pred_bboxes["p6"].shape == (BATCH_SIZE, 4 * num_anchors, 8, 8)
    assert pred_bboxes["p7"].shape == (BATCH_SIZE, 4 * num_anchors, 4, 4)


def test_reshape_retina_head_into_N_KWA_K(init_512x512_dummy_data):
    data = init_512x512_dummy_data
    num_anchors = 9
    num_classes = 20

    base = resnet50(out_features=["res3", "res4", "res5"])
    backbone = retinanet_fpn_resnet()
    head = RetinaNetHead(20)

    outputs = base(data)
    C3 = outputs["res3"]
    C4 = outputs["res4"]
    C5 = outputs["res5"]

    b_outs = backbone([C3, C4, C5])
    P3 = b_outs["fpn0"]
    P4 = b_outs["fpn1"]
    P5 = b_outs["fpn2"]
    P6 = b_outs["upsample_fpn3"]
    P7 = b_outs["upsample_fpn4"]

    pred_logits, pred_bboxes = head(P3, P4, P5, P6, P7)

    reshaped_logits = [
        permute_to_N_HWA_K(pred_logits[k], num_classes) for k in pred_logits
    ]

    reshaped_bboxes = [permute_to_N_HWA_K(pred_bboxes[k], 4) for k in pred_bboxes]

    assert reshaped_logits[0].shape == (BATCH_SIZE, 64 * 64 * num_anchors, num_classes)
    assert reshaped_logits[1].shape == (BATCH_SIZE, 32 * 32 * num_anchors, num_classes)
    assert reshaped_logits[2].shape == (BATCH_SIZE, 16 * 16 * num_anchors, num_classes)
    assert reshaped_logits[3].shape == (BATCH_SIZE, 8 * 8 * num_anchors, num_classes)
    assert reshaped_logits[4].shape == (BATCH_SIZE, 4 * 4 * num_anchors, num_classes)

    assert reshaped_bboxes[0].shape == (BATCH_SIZE, 64 * 64 * num_anchors, 4)
    assert reshaped_bboxes[1].shape == (BATCH_SIZE, 32 * 32 * num_anchors, 4)
    assert reshaped_bboxes[2].shape == (BATCH_SIZE, 16 * 16 * num_anchors, 4)
    assert reshaped_bboxes[3].shape == (BATCH_SIZE, 8 * 8 * num_anchors, 4)
    assert reshaped_bboxes[4].shape == (BATCH_SIZE, 4 * 4 * num_anchors, 4)
