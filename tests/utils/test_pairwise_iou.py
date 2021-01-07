import unittest
import torch

from utils.box_utils import pairwise_iou

# Boundary box coords: (x_min, y_min, x_max, y_max)


class TestPairwiseIOU(unittest.TestCase):
    def test_pairwise_iou_on_same_box(self):
        box = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        expected = torch.tensor([1.0])

        actual = pairwise_iou(box, box)

        assert torch.allclose(actual, expected)

    def test_pairwise_iou_with_half_shifted_x_box(self):
        box1 = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        box2 = torch.tensor([[0.75, 0.5, 1.0, 1.0]])
        expected = torch.tensor([0.5])

        actual = pairwise_iou(box1, box2)

        assert torch.allclose(actual, expected)

    def test_pairwise_iou_with_half_shifted_y_box(self):
        box1 = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        box2 = torch.tensor([[0.5, 0.75, 1.0, 1.0]])
        expected = torch.tensor([0.5])

        actual = pairwise_iou(box1, box2)

        assert torch.allclose(actual, expected)

    def test_pairwise_iou_with_half_shifted_xy_box(self):
        box1 = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        box2 = torch.tensor([[0.75, 0.75, 1.0, 1.0]])
        expected = torch.tensor([0.25])

        actual = pairwise_iou(box1, box2)

        assert torch.allclose(actual, expected)

    def test_pairwise_iou_with_multi_boxes(self):
        box1 = torch.tensor(
            [
                [0.5, 0.5, 1.0, 1.0],
                [0.5, 0.5, 1.0, 1.0],
                [0.5, 0.5, 1.0, 1.0],
                [0.5, 0.5, 1.0, 1.0],
            ]
        )
        box2 = torch.tensor(
            [
                [0.5, 0.5, 1.0, 1.0],
                [0.75, 0.5, 1.0, 1.0],
                [0.5, 0.75, 1.0, 1.0],
                [0.75, 0.75, 1.0, 1.0],
            ]
        )
        expected = torch.tensor(
            [
                [1.0, 0.5, 0.5, 0.25],
                [1.0, 0.5, 0.5, 0.25],
                [1.0, 0.5, 0.5, 0.25],
                [1.0, 0.5, 0.5, 0.25],
            ]
        )

        actual = pairwise_iou(box1, box2)

        assert torch.allclose(actual, expected)