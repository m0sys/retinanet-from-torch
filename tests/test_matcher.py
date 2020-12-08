import random
import pytest
import torch

from model.backbone.resnet import ResNet50
from model.backbone.retina_meta import RetinaNetFPN50
from model.anchor_generator import AnchorBoxGenerator
from model.matcher import Matcher
from utils.box_utils import cat_boxes, pairwise_iou

BATCH_SIZE = 1


@pytest.fixture(scope="module")
def init_512x512_dummy_data():
    return torch.randn((BATCH_SIZE, 3, 512, 512))


@pytest.fixture(scope="module")
def init_dummy_target_boxes():
    return [torch.randn((random.randint(1, 7), 4)) for _ in range(BATCH_SIZE)]


def test_matcher(init_512x512_dummy_data, init_dummy_target_boxes):
    data = init_512x512_dummy_data
    target_boxes = init_dummy_target_boxes
    num_anchors = 9
    total_anchors = (
        64 * 64 * num_anchors
        + 32 * 32 * num_anchors
        + 16 * 16 * num_anchors
        + 8 * 8 * num_anchors
        + 4 * 4 * num_anchors
    )
    labels = [-1, 0, 1]
    thresh = [0.4, 0.5]

    backbone = ResNet50()
    model = RetinaNetFPN50()
    anchor_gen = AnchorBoxGenerator(
        sizes=[32.0, 64.0, 128.0, 256.0, 512.0],
        aspect_ratios=[0.5, 1.0, 2.0],
        scales=[1.0, 2 ** (1 / 3), 2 ** (2 / 3)],
        strides=[2, 2, 2, 2, 2],
    )

    anchor_matcher = Matcher(labels=labels, thresholds=thresh)

    _, C3, C4, C5 = backbone(data)
    P3, P4, P5, P6, P7 = model(C3, C4, C5)

    all_anchors = anchor_gen([P3, P4, P5, P6, P7])
    cat_anchors = cat_boxes(all_anchors)

    match_quality = pairwise_iou(target_boxes[0], cat_anchors)

    matched_idxs, anchor_labels = anchor_matcher(match_quality)

    assert matched_idxs.shape == (total_anchors,)
    assert anchor_labels.shape == (total_anchors,)
