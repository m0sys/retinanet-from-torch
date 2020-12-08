import pytest
import torch

from model.backbone.resnet import ResNet50
from model.backbone.retina_meta import RetinaNetFPN50, RetinaNetHead
from utils.shape_utils import permute_to_N_HWA_K


BATCH_SIZE = 1

@pytest.fixture(scope="module")
def init_512x512_dummy_data():
    return torch.randn((BATCH_SIZE, 3, 512, 512))


def test_retina_head(init_512x512_dummy_data):
    data = init_512x512_dummy_data
    num_anchors = 9
    num_classes = 20

    backbone = ResNet50()
    model = RetinaNetFPN50()
    head = RetinaNetHead(20)

    _, C3, C4, C5 = backbone(data)
    del data
    P3, P4, P5, P6, P7 = model(C3, C4, C5)
    del C3, C4, C5
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

    backbone = ResNet50()
    model = RetinaNetFPN50()
    head = RetinaNetHead(num_classes)

    _, C3, C4, C5 = backbone(data)
    del data
    P3, P4, P5, P6, P7 = model(C3, C4, C5)
    del C3, C4, C5
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
