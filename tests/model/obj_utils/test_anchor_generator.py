import pytest
import torch

from model.backbone.resnet import resnet50
from model.backbone.fpn import retinanet_fpn_resnet
from model.obj_utils.anchor_generator import AnchorBoxGenerator
from utils.box_utils import cat_boxes


BATCH_SIZE = 1


@pytest.fixture(scope="module")
def init_512x512_dummy_data():
    return torch.randn((BATCH_SIZE, 3, 512, 512))


def test_anchor_generator(init_512x512_dummy_data):
    data = init_512x512_dummy_data
    num_anchors = 9

    sizes = [32.0, 64.0, 128.0, 256.0, 512.0]
    scales = [1.0, 2 ** (1 / 3), 2 ** (2 / 3)]
    sizes = [[size * scale for scale in scales] for size in sizes]

    base = resnet50(out_features=["res3", "res4", "res5"])
    backbone = retinanet_fpn_resnet()
    anchor_gen = AnchorBoxGenerator(
        sizes=sizes,
        aspect_ratios=[0.5, 1.0, 2.0],
        strides=[2, 2, 2, 2, 2],
    )

    outputs = base(data)
    C3 = outputs["res3"]
    C4 = outputs["res4"]
    C5 = outputs["res5"]

    b_outs = backbone([C3, C4, C5])
    P3 = b_outs["fpn0"]
    P4 = b_outs["fpn1"]
    P5 = b_outs["fpn2"]
    P6 = b_outs["upsample_fpn3"]
    P7 = b_outs["upsample_fpn4"]

    all_anchors = anchor_gen([P3, P4, P5, P6, P7])

    assert len(all_anchors) == 5
    assert all_anchors[0].shape == (64 * 64 * num_anchors, 4)
    assert all_anchors[1].shape == (32 * 32 * num_anchors, 4)
    assert all_anchors[2].shape == (16 * 16 * num_anchors, 4)
    assert all_anchors[3].shape == (8 * 8 * num_anchors, 4)
    assert all_anchors[4].shape == (4 * 4 * num_anchors, 4)


def test_cat_anchor_boxes(init_512x512_dummy_data):
    data = init_512x512_dummy_data
    num_anchors = 9
    total_anchors = (
        64 * 64 * num_anchors
        + 32 * 32 * num_anchors
        + 16 * 16 * num_anchors
        + 8 * 8 * num_anchors
        + 4 * 4 * num_anchors
    )

    base = resnet50(out_features=["res3", "res4", "res5"])
    backbone = retinanet_fpn_resnet()

    sizes = [32.0, 64.0, 128.0, 256.0, 512.0]
    scales = [1.0, 2 ** (1 / 3), 2 ** (2 / 3)]
    sizes = [[size * scale for scale in scales] for size in sizes]
    anchor_gen = AnchorBoxGenerator(
        sizes=sizes,
        aspect_ratios=[0.5, 1.0, 2.0],
        strides=[2, 2, 2, 2, 2],
    )

    outputs = base(data)
    C3 = outputs["res3"]
    C4 = outputs["res4"]
    C5 = outputs["res5"]

    b_outs = backbone([C3, C4, C5])
    P3 = b_outs["fpn0"]
    P4 = b_outs["fpn1"]
    P5 = b_outs["fpn2"]
    P6 = b_outs["upsample_fpn3"]
    P7 = b_outs["upsample_fpn4"]

    all_anchors = anchor_gen([P3, P4, P5, P6, P7])

    cat_anchors = cat_boxes(all_anchors)

    assert cat_anchors.shape == (total_anchors, 4)


def test_detectron2__default_anchor_generator():
    sizes = [[32.0, 64.0]]
    aspect_ratios = [[0.25, 1.0, 4.0]]

    anchor_generator = AnchorBoxGenerator(
        sizes=sizes, aspect_ratios=aspect_ratios, strides=[4], offset=0.0
    )

    # only the last two dimensions of features matter here
    num_images = 2
    features = {"stage3": torch.rand(num_images, 96, 1, 2)}
    anchors = anchor_generator([features["stage3"]])
    expected_anchor_tensor = torch.tensor(
        [
            [-32.0, -8.0, 32.0, 8.0],
            [-16.0, -16.0, 16.0, 16.0],
            [-8.0, -32.0, 8.0, 32.0],
            [-64.0, -16.0, 64.0, 16.0],
            [-32.0, -32.0, 32.0, 32.0],
            [-16.0, -64.0, 16.0, 64.0],
            [-28.0, -8.0, 36.0, 8.0],  # -28.0 == -32.0 + STRIDE (4)
            [-12.0, -16.0, 20.0, 16.0],
            [-4.0, -32.0, 12.0, 32.0],
            [-60.0, -16.0, 68.0, 16.0],
            [-28.0, -32.0, 36.0, 32.0],
            [-12.0, -64.0, 20.0, 64.0],
        ]
    )

    assert torch.allclose(anchors[0], expected_anchor_tensor)


def test_default_anchor_generator_centered():
    # test explicit args
    anchor_generator = AnchorBoxGenerator(
        sizes=[32, 64], aspect_ratios=[0.25, 1, 4], strides=[4]
    )

    # only the last two dimensions of features matter here
    num_images = 2
    features = {"stage3": torch.rand(num_images, 96, 1, 2)}
    expected_anchor_tensor = torch.tensor(
        [
            [-30.0, -6.0, 34.0, 10.0],
            [-14.0, -14.0, 18.0, 18.0],
            [-6.0, -30.0, 10.0, 34.0],
            [-62.0, -14.0, 66.0, 18.0],
            [-30.0, -30.0, 34.0, 34.0],
            [-14.0, -62.0, 18.0, 66.0],
            [-26.0, -6.0, 38.0, 10.0],
            [-10.0, -14.0, 22.0, 18.0],
            [-2.0, -30.0, 14.0, 34.0],
            [-58.0, -14.0, 70.0, 18.0],
            [-26.0, -30.0, 38.0, 34.0],
            [-10.0, -62.0, 22.0, 66.0],
        ]
    )

    anchors = anchor_generator([features["stage3"]])
    assert torch.allclose(anchors[0], expected_anchor_tensor)