import pytest
import torch

from model.backbone.resnet import ResNet50
from model.backbone.retina_meta import RetinaNetFPN50
from model.anchor_generator import AnchorBoxGenerator


@pytest.fixture(scope="module")
def init_512x512_dummy_data():
    return torch.randn((64, 3, 512, 512))


def test_anchor_generator(init_512x512_dummy_data):
    data = init_512x512_dummy_data
    num_anchors = 9

    backbone = ResNet50()
    model = RetinaNetFPN50()
    anchor_gen = AnchorBoxGenerator(
        sizes=[32.0, 64.0, 128.0, 256.0, 512.0],
        aspect_ratios=[0.5, 1.0, 2.0],
        scales=[1.0, 2 ** (1 / 3), 2 ** (2 / 3)],
        strides=[2, 2, 2, 2, 2],
    )

    _, C3, C4, C5 = backbone(data)
    P3, P4, P5, P6, P7 = model(C3, C4, C5)

    all_anchors = anchor_gen([P3, P4, P5, P6, P7])

    assert len(all_anchors) == 5
    assert all_anchors[0].shape == (64 * 64 * num_anchors, 4)
    assert all_anchors[1].shape == (32 * 32 * num_anchors, 4)
    assert all_anchors[2].shape == (16 * 16 * num_anchors, 4)
    assert all_anchors[3].shape == (8 * 8 * num_anchors, 4)
    assert all_anchors[4].shape == (4 * 4 * num_anchors, 4)
