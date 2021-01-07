import random
import pytest
import torch

from model.backbone.resnet import resnet50
from model.backbone.fpn import retinanet_fpn_resnet
from model.obj_utils.anchor_generator import AnchorBoxGenerator
from model.obj_utils.matcher import Matcher
from utils.box_utils import cat_boxes, pairwise_iou

BATCH_SIZE = 1


@pytest.fixture(scope="module")
def init_512x512_dummy_data():
    return torch.randn((BATCH_SIZE, 3, 512, 512))


@pytest.fixture(scope="module")
def init_dummy_target_boxes():
    return [torch.randn((random.randint(1, 7), 4)) for _ in range(BATCH_SIZE)]


## @pytest.mark.skip("We skip")
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

    anchor_matcher = Matcher(labels=labels, thresholds=thresh)

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

    match_quality = pairwise_iou(target_boxes[0], cat_anchors)

    matched_idxs, anchor_labels = anchor_matcher(match_quality)

    assert matched_idxs.shape == (total_anchors,)
    assert anchor_labels.shape == (total_anchors,)


def test_detectron2_matcher_test():
    anchor_matcher = Matcher([0.3, 0.7], [0, -1, 1], allow_low_quality_matches=True)
    match_quality_matrix = torch.tensor(
        [[0.15, 0.45, 0.2, 0.6], [0.3, 0.65, 0.05, 0.1], [0.05, 0.4, 0.25, 0.4]]
    )
    expected_matches = torch.tensor([1, 1, 2, 0])
    expected_match_labels = torch.tensor([-1, 1, 0, 1], dtype=torch.int8)

    matches, match_labels = anchor_matcher(match_quality_matrix)
    assert torch.allclose(matches, expected_matches)
    assert torch.allclose(match_labels, expected_match_labels)
