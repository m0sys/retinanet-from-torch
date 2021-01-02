import pytest
import torch

from model.backbone.resnet import resnet50
from model.backbone.retina_meta import RetinaNetFPN50

BATCH_SIZE = 1


@pytest.fixture(scope="module")
def init_512x512_dummy_data():
    return torch.randn((BATCH_SIZE, 3, 512, 512))


def test_retina_fpn(init_512x512_dummy_data):
    data = init_512x512_dummy_data
    backbone = resnet50(out_features=["res3", "res4", "res5"])
    model = RetinaNetFPN50()

    outputs = backbone(data)
    C3 = outputs["res3"]
    C4 = outputs["res4"]
    C5 = outputs["res5"]

    P3, P4, P5, P6, P7 = model(C3, C4, C5)

    assert P3.shape == (BATCH_SIZE, 256, 64, 64)
    assert P4.shape == (BATCH_SIZE, 256, 32, 32)
    assert P5.shape == (BATCH_SIZE, 256, 16, 16)
    assert P6.shape == (BATCH_SIZE, 256, 8, 8)
    assert P7.shape == (BATCH_SIZE, 256, 4, 4)
