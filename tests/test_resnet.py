import pdb
import pytest
import torch

from model.backbone.resnet import ResNet50, resnet50, resnet101, resnet152


BATCH_SIZE = 1


@pytest.fixture(scope="module")
def init_224x224_dummy_data():
    return torch.randn((BATCH_SIZE, 3, 224, 224))


@pytest.fixture(scope="module")
def init_512x512_dummy_data():
    return torch.randn((BATCH_SIZE, 3, 512, 512))


def test_resnet50_224__wo_classification(init_224x224_dummy_data):
    data = init_224x224_dummy_data
    model = ResNet50()

    C2, C3, C4, C5 = model(data)
    assert C2.shape == (BATCH_SIZE, 256, 56, 56)
    assert C3.shape == (BATCH_SIZE, 512, 28, 28)
    assert C4.shape == (BATCH_SIZE, 1024, 14, 14)
    assert C5.shape == (BATCH_SIZE, 2048, 7, 7)


def test_resnet50_512__wo_classification(init_512x512_dummy_data):
    data = init_512x512_dummy_data
    model = ResNet50()

    C2, C3, C4, C5 = model(data)
    assert C2.shape == (BATCH_SIZE, 256, 128, 128)
    assert C3.shape == (BATCH_SIZE, 512, 64, 64)
    assert C4.shape == (BATCH_SIZE, 1024, 32, 32)
    assert C5.shape == (BATCH_SIZE, 2048, 16, 16)


def test_resnet50_224__w_classification(init_224x224_dummy_data):
    data = init_224x224_dummy_data
    model = ResNet50(num_classes=1000)

    out = model(data)
    assert out.shape == (BATCH_SIZE, 1000)


def test_resnet_with_50_layers_224_image_wo_classification(init_224x224_dummy_data):
    data = init_224x224_dummy_data

    model = resnet50(out_features=["res2", "res3", "res4", "res5"])

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    ## pdb.set_trace()
    assert len(model.state_dict().items()) == 330 - 12


def test_resnet_with_101_layers_224_image_wo_classification(init_224x224_dummy_data):
    data = init_224x224_dummy_data

    model = resnet101(out_features=["res2", "res3", "res4", "res5"])

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 636 - 12


def test_resnet_with_152_layers_224_image_wo_classification(init_224x224_dummy_data):
    data = init_224x224_dummy_data

    model = resnet152(out_features=["res2", "res3", "res4", "res5"])

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 942 - 12


def test_resnet_with_50_layers_224_image_w_classification(init_224x224_dummy_data):
    data = init_224x224_dummy_data

    model = resnet50(out_features=["res2", "res3", "res4", "res5"], num_classes=1000)

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 332 - 12


def test_resnet_with_101_layers_224_image_w_classification(init_224x224_dummy_data):
    data = init_224x224_dummy_data

    model = resnet101(out_features=["res2", "res3", "res4", "res5"], num_classes=1000)

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 638 - 12


def test_resnet_with_152_layers_224_image_w_classification(init_224x224_dummy_data):
    data = init_224x224_dummy_data

    model = resnet152(out_features=["res2", "res3", "res4", "res5"], num_classes=1000)

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 944 - 12
