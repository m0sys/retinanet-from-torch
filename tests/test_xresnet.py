import pytest
import torch

from model.backbone.xresnet import xresnet50, xresnet101, xresnet152

BATCH_SIZE = 1


@pytest.fixture(scope="module")
def init_224x224_dummy_data():
    return torch.randn((BATCH_SIZE, 3, 224, 224))


@pytest.fixture(scope="module")
def init_512x512_dummy_data():
    return torch.randn((BATCH_SIZE, 3, 512, 512))


def test_xresnet_with_50_layers_224_image_wo_classification(init_224x224_dummy_data):
    data = init_224x224_dummy_data

    model = xresnet50(out_features=["res2", "res3", "res4", "res5"])

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 330


def test_xresnet_with_101_layers_224_image_wo_classification(init_224x224_dummy_data):
    data = init_224x224_dummy_data

    model = xresnet101(out_features=["res2", "res3", "res4", "res5"])

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 636


def test_xresnet_with_152_layers_224_image_wo_classification(init_224x224_dummy_data):
    data = init_224x224_dummy_data

    model = xresnet152(out_features=["res2", "res3", "res4", "res5"])

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 942


def test_xresnet_with_50_layers_224_image_wo_classification_w_dropout(
    init_224x224_dummy_data,
):
    data = init_224x224_dummy_data

    model = xresnet50(out_features=["res2", "res3", "res4", "res5"], use_dropout=True)

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 330


def test_xresnet_with_101_layers_224_image_wo_classification_w_dropout(
    init_224x224_dummy_data,
):
    data = init_224x224_dummy_data

    model = xresnet101(out_features=["res2", "res3", "res4", "res5"], use_dropout=True)

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 636


def test_xresnet_with_152_layers_224_image_wo_classification_w_dropout(
    init_224x224_dummy_data,
):
    data = init_224x224_dummy_data

    model = xresnet152(out_features=["res2", "res3", "res4", "res5"], use_dropout=True)

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 942


def test_xresnet_with_50_layers_224_image_w_classification(init_224x224_dummy_data):
    data = init_224x224_dummy_data

    model = xresnet50(out_features=["res2", "res3", "res4", "res5"], num_classes=1000)

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 332


def test_xresnet_with_101_layers_224_image_w_classification(init_224x224_dummy_data):
    data = init_224x224_dummy_data

    model = xresnet101(out_features=["res2", "res3", "res4", "res5"], num_classes=1000)

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 638


def test_xresnet_with_152_layers_224_image_w_classification(init_224x224_dummy_data):
    data = init_224x224_dummy_data

    model = xresnet152(out_features=["res2", "res3", "res4", "res5"], num_classes=1000)

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 944


def test_xresnet_with_50_layers_224_image_w_classification_and_dropout(
    init_224x224_dummy_data,
):
    data = init_224x224_dummy_data

    model = xresnet50(
        out_features=["res2", "res3", "res4", "res5"],
        num_classes=1000,
        use_dropout=True,
    )

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 332


def test_xresnet_with_101_layers_224_image_w_classification_and_dropout(
    init_224x224_dummy_data,
):
    data = init_224x224_dummy_data

    model = xresnet101(
        out_features=["res2", "res3", "res4", "res5"],
        num_classes=1000,
        use_dropout=True,
    )

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 638


def test_xresnet_with_152_layers_224_image_w_classification_and_dropout(
    init_224x224_dummy_data,
):
    data = init_224x224_dummy_data

    model = xresnet152(
        out_features=["res2", "res3", "res4", "res5"],
        num_classes=1000,
        use_dropout=True,
    )

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 944