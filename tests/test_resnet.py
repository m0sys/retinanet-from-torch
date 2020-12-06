import pytest
import torch

from model.backbone.resnet import ResNet50


@pytest.fixture(scope="module")
def init_224x224_dummy_data():
    return torch.randn((64, 3, 224, 224))


@pytest.fixture(scope="module")
def init_512x512_dummy_data():
    return torch.randn((64, 3, 512, 512))


def test_resnet50_224__wo_classification(init_224x224_dummy_data):
    data = init_224x224_dummy_data
    model = ResNet50()

    C2, C3, C4, C5 = model(data)
    assert C2.shape == (64, 256, 56, 56)
    assert C3.shape == (64, 512, 28, 28)
    assert C4.shape == (64, 1024, 14, 14)
    assert C5.shape == (64, 2048, 7, 7)


def test_resnet50_512__wo_classification(init_512x512_dummy_data):
    data = init_512x512_dummy_data
    model = ResNet50()

    C2, C3, C4, C5 = model(data)
    assert C2.shape == (64, 256, 128, 128)
    assert C3.shape == (64, 512, 64, 64)
    assert C4.shape == (64, 1024, 32, 32)
    assert C5.shape == (64, 2048, 16, 16)


def test_resnet50_224__w_classification(init_224x224_dummy_data):
    data = init_224x224_dummy_data
    model = ResNet50(num_classes=1000)

    out = model(data)
    assert out.shape == (64, 1000)