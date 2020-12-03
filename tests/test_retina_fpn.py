import pytest
import torch

from model.backbone.resnet import ResNet50
from model.backbone.retina_meta import RetinaNetFPN50


@pytest.fixture(scope="module")
def init_512x512_dummy_data():
    return torch.randn((64, 3, 512, 512))


def test_retina_fpn(init_512x512_dummy_data):
    data = init_512x512_dummy_data
    backbone = ResNet50()
    model = RetinaNetFPN50()

    _, C3, C4, C5 = backbone(data)
    P3, P4, P5, P6, P7 = model(C3, C4, C5)

    assert P3.shape == (64, 256, 64, 64)
    assert P4.shape == (64, 256, 32, 32)
    assert P5.shape == (64, 256, 16, 16)
    assert P6.shape == (64, 256, 8, 8)
    assert P7.shape == (64, 256, 4, 4)
