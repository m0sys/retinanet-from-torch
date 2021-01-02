from pathlib import Path
import pytest
from PIL import Image
import torch
from torchvision import transforms

from model.backbone.resnet import resnet50, resnet101, resnet152


BATCH_SIZE = 1
ROOT_IMG_PATH = Path("./notebooks/images")
TRANSFORM = transforms.Compose(
    [  # [1]
        transforms.Resize(256),  # [2]
        transforms.CenterCrop(224),  # [3]
        transforms.ToTensor(),  # [4]
        transforms.Normalize(  # [5]
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # [6]  # [7]
        ),
    ]
)
with open("notebooks/imagenet_classes.txt") as f:
    IDX_2_LABEL = eval(f.read())


@pytest.fixture(scope="module")
def init_224x224_dummy_data():
    return torch.randn((BATCH_SIZE, 3, 224, 224))


@pytest.fixture(scope="module")
def init_512x512_dummy_data():
    return torch.randn((BATCH_SIZE, 3, 512, 512))


def test_resnet_with_50_layers_224_image_wo_classification(init_224x224_dummy_data):
    data = init_224x224_dummy_data

    model = resnet50(out_features=["res2", "res3", "res4", "res5"])

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

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


def test_pretrained_resnet_with_50_layers_224_image_w_classification(
    init_224x224_dummy_data,
):
    data = init_224x224_dummy_data

    model = resnet50(
        out_features=["res2", "res3", "res4", "res5"], num_classes=1000, pretrained=True
    )

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 332 - 12


def test_pretrained_resnet_with_101_layers_224_image_w_classification(
    init_224x224_dummy_data,
):
    data = init_224x224_dummy_data

    model = resnet101(
        out_features=["res2", "res3", "res4", "res5"], num_classes=1000, pretrained=True
    )

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 638 - 12


def test_pretrained_resnet_with_152_layers_224_image_w_classification(
    init_224x224_dummy_data,
):
    data = init_224x224_dummy_data

    model = resnet152(
        out_features=["res2", "res3", "res4", "res5"], num_classes=1000, pretrained=True
    )

    outputs = model(data)

    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 944 - 12


def test_pretrained_inferance_resnet_with_50_layers_224_image_w_classification(
    init_224x224_dummy_data,
):
    data = init_224x224_dummy_data

    model = resnet50(
        out_features=["res2", "res3", "res4", "res5", "fc"],
        num_classes=1000,
        pretrained=True,
    )

    outputs = model(data)

    model.eval()
    img = Image.open(ROOT_IMG_PATH / "languar.jpg")
    pred = _make_prediction(model, img, IDX_2_LABEL)

    assert pred == "langur"
    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 332 - 12


def test_pretrained_inferance_resnet_with_101_layers_224_image_w_classification(
    init_224x224_dummy_data,
):
    data = init_224x224_dummy_data

    model = resnet101(
        out_features=["res2", "res3", "res4", "res5", "fc"],
        num_classes=1000,
        pretrained=True,
    )

    outputs = model(data)

    model.eval()
    img = Image.open(ROOT_IMG_PATH / "languar.jpg")
    pred = _make_prediction(model, img, IDX_2_LABEL)

    assert pred == "langur"
    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 638 - 12


def test_pretrained_inferance_resnet_with_152_layers_224_image_w_classification(
    init_224x224_dummy_data,
):
    data = init_224x224_dummy_data

    model = resnet152(
        out_features=["res2", "res3", "res4", "res5", "fc"],
        num_classes=1000,
        pretrained=True,
    )

    outputs = model(data)

    model.eval()
    img = Image.open(ROOT_IMG_PATH / "languar.jpg")
    pred = _make_prediction(model, img, IDX_2_LABEL)

    assert pred == "langur"
    assert outputs["res2"].shape == (BATCH_SIZE, 256, 56, 56)
    assert outputs["res3"].shape == (BATCH_SIZE, 512, 28, 28)
    assert outputs["res4"].shape == (BATCH_SIZE, 1024, 14, 14)
    assert outputs["res5"].shape == (BATCH_SIZE, 2048, 7, 7)

    assert len(model.state_dict().items()) == 944 - 12


def _make_prediction(model, img, label_dict):
    img_t = TRANSFORM(img)
    batch_t = torch.unsqueeze(img_t, 0)
    out = model(batch_t)
    _, index = torch.max(out["fc"], 1)
    return label_dict[index.item()]
