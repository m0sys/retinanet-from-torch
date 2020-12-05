"""Utility functions for bounding box gymnastics."""

from typing import List
import torch
from torch import Tensor


def cat_boxes(boxes_list: List[Tensor]):
    """
    Concatenates a list of Boxes (HW x 4 tensors) into a single tensor.
    """
    if len(boxes_list) == 0:
        return torch.empty(0)

    cat_boxes = torch.cat([b for b in boxes_list], dim=0)
    return cat_boxes


def pairwise_iou(set_1, set_2):
    """
    Find the IOU (intersection over union) of every combination between two
    sets of boxes that are in boundary containers.
    Args:
        set_1: a tensor of dimensions (n1, 4)
        set_2: a tensor of dimensions (n2, 4)
    Returns:
        a tensor of dimensions (n1, n2)
    """
    intersection = pairwise_intersection(set_1, set_2)  # (n1, n2)

    areas_set_1 = _find_area(set_1)  # (n1)
    areas_set_2 = _find_area(set_2)  # (n2)

    union = (
        areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection
    )  # (n1, n2)

    return intersection / union  # (n1, n2)


def pairwise_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of
    boxes that are in boundary coordinates.
    Args:
        set_1: a tensor of dimensions (n1, 4)
        set_2: a tensor of dimensions (n2, 4)
    Returns:
        a tensor of dimensions (n1, n2)
    """

    lower_bounds = torch.max(
        set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0)
    )  # (n1, n2, 2)
    upper_bounds = torch.min(
        set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0)
    )  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def _find_area(sset):
    """
    Find area of boxes that are in boundary container.
    Args:
        sset: a tensor of dimensions (n, 4)
    Returns:
        a tensor of dimensions (n)
    """

    return (sset[:, 2] - sset[:, 0]) * (sset[:, 3] - sset[:, 1])