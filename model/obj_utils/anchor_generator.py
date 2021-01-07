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
    """
    Compute anchors in the standard way described in
    "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
    paper.
    """

    def __init__(
        self,
        sizes: Optional[Union[List[List[float]], List[float]]] = None,
        aspect_ratios: Optional[Union[List[List[float]], List[float]]] = None,
        scales: Optional[Union[List[List[float]], List[float]]] = None,
        strides: Optional[List[int]] = None,
        pyramid_levels=[3, 4, 5, 6, 7],
        offset: float = 0.5,
    ):
        """
        Args:
            sizes: If sizes is a list[list[float]], sizes[i] is the list of anchor
                sizes. (i.e. sqrt of anchor area) to use for the i-th feature map.
                If sizes is list[float], the sizes are used for all feature maps.
                Anchor sizes are given in absolute lengths in units of the input image; they do not dynamically scale if the input image size changes.
            aspect_ratios: list of aspect ratios (i.e. height / width) to use for
                anchors. Same "broadcast" rule for `sizes` applies.
            strides: stride of each input feature.
            offset: Relative offset between the center of the first anchor and
                the top-left corner of the image. Value has to be in [0, 1).
                Recommend to use 0.5, which means half stride.
        """
        super().__init__()

        if strides is None:
            self.strides = [2 ** x for x in pyramid_levels]
        else:
            self.strides = strides

        if scales is None:
            scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

        if sizes is None:
            sizes = [2 ** (x + 2) for x in pyramid_levels]
            sizes = [[size * scale for scale in scales] for size in sizes]
        if aspect_ratios is None:
            aspect_ratios = [0.5, 1.0, 2.0]

        self.num_features = len(self.strides)
        sizes = _broadcast_params(sizes, self.num_features)
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

    @staticmethod
    def generate_anchor_boxes(sizes=(32, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
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
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)

        return anchors_over_all_feature_maps

    def _grid_anchors(self, grid_sizes: List[List[int]]):
        """
        Shift canonical anchors to build a grid of anchor boxes.

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

    @staticmethod
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
