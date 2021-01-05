import pytest
import torch

from model.model import retina_resnet50, retina_resnet101, retina_resnet152

BATCH_SIZE = 1


@pytest.fixture(scope="module")
def init_512x512_dummy_data():
    return torch.randn((BATCH_SIZE, 3, 512, 512))


def test_retina_resnet50(init_512x512_dummy_data):
    num_anchors = 9
    num_classes = 20
    data = init_512x512_dummy_data
    model = retina_resnet50(num_classes)

    reshaped_logits, reshaped_bboxes, _ = model(data)
    assert reshaped_logits[0].shape == (BATCH_SIZE, 64 * 64 * num_anchors, num_classes)
    assert reshaped_logits[1].shape == (BATCH_SIZE, 32 * 32 * num_anchors, num_classes)
    assert reshaped_logits[2].shape == (BATCH_SIZE, 16 * 16 * num_anchors, num_classes)
    assert reshaped_logits[3].shape == (BATCH_SIZE, 8 * 8 * num_anchors, num_classes)
    assert reshaped_logits[4].shape == (BATCH_SIZE, 4 * 4 * num_anchors, num_classes)

    assert reshaped_bboxes[0].shape == (BATCH_SIZE, 64 * 64 * num_anchors, 4)
    assert reshaped_bboxes[1].shape == (BATCH_SIZE, 32 * 32 * num_anchors, 4)
    assert reshaped_bboxes[2].shape == (BATCH_SIZE, 16 * 16 * num_anchors, 4)
    assert reshaped_bboxes[3].shape == (BATCH_SIZE, 8 * 8 * num_anchors, 4)
    assert reshaped_bboxes[4].shape == (BATCH_SIZE, 4 * 4 * num_anchors, 4)
