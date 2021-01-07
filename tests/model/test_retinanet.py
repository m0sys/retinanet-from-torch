import pytest
import torch

from model.model import retina_resnet50, retina_resnet101, retina_resnet152

BATCH_SIZE = 1


@pytest.fixture(scope="module")
def init_512x512_dummy_data():
    return torch.randn((BATCH_SIZE, 3, 512, 512))


def test_retina_resnet50_out_shapes(init_512x512_dummy_data):
    num_anchors = 9
    num_classes = 20
    data = init_512x512_dummy_data
    model = retina_resnet50(num_classes)

    outputs = model(data)
    pred_logits = outputs["pred_logits"]
    pred_bboxes = outputs["pred_bboxes"]
    anchors = outputs["anchors"]

    assert pred_logits[0].shape == (BATCH_SIZE, 64 * 64 * num_anchors, num_classes)
    assert pred_logits[1].shape == (BATCH_SIZE, 32 * 32 * num_anchors, num_classes)
    assert pred_logits[2].shape == (BATCH_SIZE, 16 * 16 * num_anchors, num_classes)
    assert pred_logits[3].shape == (BATCH_SIZE, 8 * 8 * num_anchors, num_classes)
    assert pred_logits[4].shape == (BATCH_SIZE, 4 * 4 * num_anchors, num_classes)

    assert pred_bboxes[0].shape == (BATCH_SIZE, 64 * 64 * num_anchors, 4)
    assert pred_bboxes[1].shape == (BATCH_SIZE, 32 * 32 * num_anchors, 4)
    assert pred_bboxes[2].shape == (BATCH_SIZE, 16 * 16 * num_anchors, 4)
    assert pred_bboxes[3].shape == (BATCH_SIZE, 8 * 8 * num_anchors, 4)
    assert pred_bboxes[4].shape == (BATCH_SIZE, 4 * 4 * num_anchors, 4)

    assert anchors[0].shape == (64 * 64 * num_anchors, 4)
    assert anchors[1].shape == (32 * 32 * num_anchors, 4)
    assert anchors[2].shape == (16 * 16 * num_anchors, 4)
    assert anchors[3].shape == (8 * 8 * num_anchors, 4)
    assert anchors[4].shape == (4 * 4 * num_anchors, 4)


def test_retina_resnet101_out_shapes(init_512x512_dummy_data):
    num_anchors = 9
    num_classes = 20
    data = init_512x512_dummy_data
    model = retina_resnet101(num_classes)

    outputs = model(data)
    pred_logits = outputs["pred_logits"]
    pred_bboxes = outputs["pred_bboxes"]
    anchors = outputs["anchors"]

    assert pred_logits[0].shape == (BATCH_SIZE, 64 * 64 * num_anchors, num_classes)
    assert pred_logits[1].shape == (BATCH_SIZE, 32 * 32 * num_anchors, num_classes)
    assert pred_logits[2].shape == (BATCH_SIZE, 16 * 16 * num_anchors, num_classes)
    assert pred_logits[3].shape == (BATCH_SIZE, 8 * 8 * num_anchors, num_classes)
    assert pred_logits[4].shape == (BATCH_SIZE, 4 * 4 * num_anchors, num_classes)

    assert pred_bboxes[0].shape == (BATCH_SIZE, 64 * 64 * num_anchors, 4)
    assert pred_bboxes[1].shape == (BATCH_SIZE, 32 * 32 * num_anchors, 4)
    assert pred_bboxes[2].shape == (BATCH_SIZE, 16 * 16 * num_anchors, 4)
    assert pred_bboxes[3].shape == (BATCH_SIZE, 8 * 8 * num_anchors, 4)
    assert pred_bboxes[4].shape == (BATCH_SIZE, 4 * 4 * num_anchors, 4)

    assert anchors[0].shape == (64 * 64 * num_anchors, 4)
    assert anchors[1].shape == (32 * 32 * num_anchors, 4)
    assert anchors[2].shape == (16 * 16 * num_anchors, 4)
    assert anchors[3].shape == (8 * 8 * num_anchors, 4)
    assert anchors[4].shape == (4 * 4 * num_anchors, 4)


def test_retina_resnet152_out_shapes(init_512x512_dummy_data):
    num_anchors = 9
    num_classes = 20
    data = init_512x512_dummy_data
    model = retina_resnet152(num_classes)

    outputs = model(data)
    pred_logits = outputs["pred_logits"]
    pred_bboxes = outputs["pred_bboxes"]
    anchors = outputs["anchors"]

    assert pred_logits[0].shape == (BATCH_SIZE, 64 * 64 * num_anchors, num_classes)
    assert pred_logits[1].shape == (BATCH_SIZE, 32 * 32 * num_anchors, num_classes)
    assert pred_logits[2].shape == (BATCH_SIZE, 16 * 16 * num_anchors, num_classes)
    assert pred_logits[3].shape == (BATCH_SIZE, 8 * 8 * num_anchors, num_classes)
    assert pred_logits[4].shape == (BATCH_SIZE, 4 * 4 * num_anchors, num_classes)

    assert pred_bboxes[0].shape == (BATCH_SIZE, 64 * 64 * num_anchors, 4)
    assert pred_bboxes[1].shape == (BATCH_SIZE, 32 * 32 * num_anchors, 4)
    assert pred_bboxes[2].shape == (BATCH_SIZE, 16 * 16 * num_anchors, 4)
    assert pred_bboxes[3].shape == (BATCH_SIZE, 8 * 8 * num_anchors, 4)
    assert pred_bboxes[4].shape == (BATCH_SIZE, 4 * 4 * num_anchors, 4)

    assert anchors[0].shape == (64 * 64 * num_anchors, 4)
    assert anchors[1].shape == (32 * 32 * num_anchors, 4)
    assert anchors[2].shape == (16 * 16 * num_anchors, 4)
    assert anchors[3].shape == (8 * 8 * num_anchors, 4)
    assert anchors[4].shape == (4 * 4 * num_anchors, 4)