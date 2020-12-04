from typing import Union, List, Tuple, Optional
import math

import torch
import torch.nn as nn


class BufferList(nn.Module):
    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(i), buffer)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())

    def __repr__(self):
        return str(self._buffers.values())


def _broadcast_params(params: Union[List[float], Tuple[float]], num_features: int):
    if not isinstance(params[0], (list, tuple)):
        return [params] * num_features
    if len(params) == 1:
        return list(params) * num_features

    assert len(params) == num_features
    return params


class AnchorBoxGenerator(nn.Module):
    def __init__(
        self,
        sizes: List[float],
        aspect_ratios: List[float],
        strides: List[int],
        scales: Optional[List[float]] = [1.0],
        offset: Optional[float] = 0.5,
    ):
        """
        Compute anchors in the standard way described in
        "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
        paper.
        """
        super().__init__()

        self.strides = strides
        self.num_features = len(self.strides)
        sizes = _broadcast_params(
            [[size * scale for scale in scales] for size in sizes], self.num_features
        )
        aspect_ratios = _broadcast_params(aspect_ratios, self.num_features)
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios)

        self.offset = offset
        assert 0.0 <= self.offset < 1.0

    def _calculate_anchors(self, sizes, aspect_ratios):
        cell_anchors = [
            self.generate_anchor_boxes(s, a).float()
            for s, a in zip(sizes, aspect_ratios)
        ]
        return BufferList(cell_anchors)

    def generate_anchor_boxes(
        self, sizes=(32, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
    ):
        """
        Generate a tensor storing canonical anchor boxes of different
        sizes and aspect ratios centered at (0, 0).

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes in
            bounding-box (X_min, Y_min, X_max, Y_max) coords.
        """

        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = (
                    -w / 2.0,
                    -h / 2.0,
                    w / 2.0,
                    h / 2.0,
                )  # centered @ (0, 0)
                anchors.append([x0, y0, x1, y1])

        return torch.tensor(anchors)

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features: list of backbone feature maps on which to generate anchors.

        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)

        return anchors_over_all_feature_maps

    def _grid_anchors(self, grid_sizes: List[List[int]]):
        """
        Returns:
            list[Tensor]: #feature map tensors of shape (locs x cell_anchors) * 4
        """
        anchors = []
        buffers = [x[1] for x in self.cell_anchors.named_buffers()]

        for size, stride, base_anchors in zip(grid_sizes, self.strides, buffers):
            shift_x, shift_y = self._create_grid_offsets(
                size, stride, self.offset, base_anchors.device
            )
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def _create_grid_offsets(
        size: List[int], stride: int, offset: float, device: torch.device
    ):
        grid_height, grid_width = size
        shifts_x = torch.arange(
            offset * stride,
            grid_width * stride,
            step=stride,
            dtype=torch.float32,
            device=device,
        )

        shifts_y = torch.arange(
            offset * stride,
            grid_height * stride,
            step=stride,
            dtype=torch.float32,
            device=device,
        )

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        return shift_x, shift_y
