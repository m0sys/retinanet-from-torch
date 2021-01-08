"""Utility functions for bounding box gymnastics."""

## import pdb
from typing import List, Union
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

    areas_set_1 = find_area(set_1)  # (n1)
    areas_set_2 = find_area(set_2)  # (n2)

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

    ## pdb.set_trace(header="box_utils.py -> pairwise_intersection -> beginning")
    lower_bounds = torch.max(
        set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0)
    )  # (n1, n2, 2)
    upper_bounds = torch.min(
        set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0)
    )  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_area(sset):
    """
    Find area of boxes that are in boundary container.
    Args:
        sset: a tensor of dimensions (n, 4)
    Returns:
        a tensor of dimensions (n)
    """

    return (sset[:, 2] - sset[:, 0]) * (sset[:, 3] - sset[:, 1])


def remove_zero_area_bboxes(
    bs, bboxes: Union[List[Tensor], Tensor], lbls: Union[List[Tensor], Tensor]
):
    """
    Remove any bboxes in a given batch if the area is zero.

    This helps training by avoiding the accidental inclusion of bboxes that
    have zero area which can cause the loss to become unstable.

    See https://github.com/jwyang/faster-rcnn.pytorch/issues/136#issuecomment-390544655
    for more details.
    """
    t_bboxes, t_lbls = [], []
    # TODO: Can I avoid this bs loop?
    for i in range(bs):
        areas = find_area(bboxes[i])
        area_mask = areas > 0
        t_bboxes.append(bboxes[i][area_mask])
        t_lbls.append(lbls[i][area_mask])
    return t_bboxes, t_lbls
