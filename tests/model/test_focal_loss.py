import random
import pytest
import torch

from model.model import retina_resnet50
from model.loss import RetinaLoss

BATCH_SIZE = 1
NUM_CLASSES = 20
NUM_ANCHORS = 9


@pytest.fixture(scope="module")
def init_512x512_dummy_data():
    return torch.randn((BATCH_SIZE, 3, 512, 512))


@pytest.fixture(scope="module")
def init_dummy_target_boxes_and_labels():
    num_items = [random.randint(0, 7) for _ in range(BATCH_SIZE)]
    bboxes = [torch.randn((num_items[i], 4)) for i in range(BATCH_SIZE)]
    labels = [
        torch.randint(high=NUM_CLASSES, size=(num_items[i], 1))
        for i in range(BATCH_SIZE)
    ]
    return bboxes, labels


def test_retina_loss_w_retina_res50(
    init_512x512_dummy_data, init_dummy_target_boxes_and_labels
):
    data = init_512x512_dummy_data
    bboxes, labels = init_dummy_target_boxes_and_labels
    model = retina_resnet50(num_classes=NUM_CLASSES)
    crit = RetinaLoss(num_classes=NUM_CLASSES)

    outputs = model(data)
    ## print(
    ##     f"Pred Logits: \n\n {outputs['pred_logits']}, Pred bboxes: \n\n{outputs['pred_bboxes']}"
    ## )

    loss = crit(outputs, bboxes, labels)

    ## print(loss)