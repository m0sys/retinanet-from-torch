"""This module contains general testing utils."""

import torch


def random_boxes(num_boxes, max_coord=100, device="cpu"):
    """
    Create a random Nx4 boxes tensor, with coordinates < max_coord.
    """
    boxes = torch.rand(num_boxes, 4, device=device) * (max_coord * 0.5)
    boxes.clamp_(min=1.0)  # tiny boxes cause numerical instability in box regression
    # Note: the implementation of this function in torchvision is:
    # boxes[:, 2:] += torch.rand(N, 2) * 100
    # but it does not guarantee non-negative widths/heights constraints:
    # boxes[:, 2] >= boxes[:, 0] and boxes[:, 3] >= boxes[:, 1]:
    boxes[:, 2:] += boxes[:, :2]
    return boxes