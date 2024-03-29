import pytest
import torch

from model.backbone.resnet import resnet50
from model.backbone.fpn import FPN, retinanet_fpn_resnet

BATCH_SIZE = 1


@pytest.fixture(scope="module")
def init_512x512_dummy_data():
    return torch.randn((BATCH_SIZE, 3, 512, 512))


def test_bare_fpn(init_512x512_dummy_data):
    data = init_512x512_dummy_data
    base = resnet50(out_features=["res3", "res4", "res5"])
    backbone = FPN(upsample_stages=[2048, 256], downsample_stages=[2048, 1024, 512])

    outputs = base(data)
    C3 = outputs["res3"]
    C4 = outputs["res4"]
    C5 = outputs["res5"]

    outputs = backbone([C3, C4, C5])
    P3 = outputs["fpn0"]
    P4 = outputs["fpn1"]
    P5 = outputs["fpn2"]
    P6 = outputs["upsample_fpn3"]
    P7 = outputs["upsample_fpn4"]

    assert P3.shape == (BATCH_SIZE, 256, 64, 64)
    assert P4.shape == (BATCH_SIZE, 256, 32, 32)
    assert P5.shape == (BATCH_SIZE, 256, 16, 16)
    assert P6.shape == (BATCH_SIZE, 256, 8, 8)
    assert P7.shape == (BATCH_SIZE, 256, 4, 4)


def test_retina_fpn_resnet(init_512x512_dummy_data):
    data = init_512x512_dummy_data
    base = resnet50(out_features=["res3", "res4", "res5"])
    backbone = retinanet_fpn_resnet()

    outputs = base(data)
    C3 = outputs["res3"]
    C4 = outputs["res4"]
    C5 = outputs["res5"]

    outputs = backbone([C3, C4, C5])
    P3 = outputs["fpn0"]
    P4 = outputs["fpn1"]
    P5 = outputs["fpn2"]
    P6 = outputs["upsample_fpn3"]
    P7 = outputs["upsample_fpn4"]

    assert P3.shape == (BATCH_SIZE, 256, 64, 64)
    assert P4.shape == (BATCH_SIZE, 256, 32, 32)
    assert P5.shape == (BATCH_SIZE, 256, 16, 16)
    assert P6.shape == (BATCH_SIZE, 256, 8, 8)
    assert P7.shape == (BATCH_SIZE, 256, 4, 4)