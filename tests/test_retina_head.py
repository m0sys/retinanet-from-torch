import pytest
import torch

from model.backbone.resnet import ResNet50
from model.backbone.retina_meta import RetinaNetFPN50, RetinaNetHead


@pytest.fixture(scope="module")
def init_512x512_dummy_data():
    return torch.randn((32, 3, 512, 512))


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

    assert pred_logits["p3"].shape == (32, num_classes * num_anchors, 64, 64)
    assert pred_logits["p4"].shape == (32, num_classes * num_anchors, 32, 32)
    assert pred_logits["p5"].shape == (32, num_classes * num_anchors, 16, 16)
    assert pred_logits["p6"].shape == (32, num_classes * num_anchors, 8, 8)
    assert pred_logits["p7"].shape == (32, num_classes * num_anchors, 4, 4)

    assert pred_bboxes["p3"].shape == (32, 4 * num_anchors, 64, 64)
    assert pred_bboxes["p4"].shape == (32, 4 * num_anchors, 32, 32)
    assert pred_bboxes["p5"].shape == (32, 4 * num_anchors, 16, 16)
    assert pred_bboxes["p6"].shape == (32, 4 * num_anchors, 8, 8)
    assert pred_bboxes["p7"].shape == (32, 4 * num_anchors, 4, 4)
