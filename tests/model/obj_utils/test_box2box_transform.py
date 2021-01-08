import unittest
import pytest
import torch


from model.obj_utils.box_regression import Box2BoxTransform
from utils.testing import random_boxes

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
BATCH_SIZE = 1


@pytest.fixture(scope="module")
def init_512x512_dummy_data():
    return torch.randn((BATCH_SIZE, 3, 512, 512))


class TestBox2BoxTransform(unittest.TestCase):
    def test_detectron2_reconstruction_test(self):
        weights = (5, 5, 10, 10)
        b2b_tfm = Box2BoxTransform(weights=weights)
        src_boxes = random_boxes(10)
        dst_boxes = random_boxes(10)

        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            src_boxes = src_boxes.to(device=device)
            dst_boxes = dst_boxes.to(device=device)
            deltas = b2b_tfm.get_deltas(src_boxes, dst_boxes)
            dst_boxes_reconstructed = b2b_tfm.apply_deltas(deltas, src_boxes)
            self.assertTrue(torch.allclose(dst_boxes, dst_boxes_reconstructed))

    @unittest.skipIf(TORCH_VERSION < (1, 8), "Insufficient pytorch version")
    def test_detectron2_apply_deltas_tracing_test(self):
        weights = (5, 5, 10, 10)
        b2b_tfm = Box2BoxTransform(weights=weights)

        with torch.no_grad():
            func = torch.jit.trace(
                b2b_tfm.apply_deltas, (torch.randn(10, 20), torch.randn(10, 4))
            )

            o = func(torch.randn(10, 20), torch.randn(10, 4))
            self.assertEqual(o.shape, (10, 20))
            o = func(torch.randn(5, 20), torch.randn(5, 4))
            self.assertEqual(o.shape, (5, 20))
