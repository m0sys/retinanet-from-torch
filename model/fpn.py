"""
This module contains the implementation for a dynamic FPN model.

For details see <paper>: https://arxiv.org/abs/1612.03144
"""
## import pdb
from typing import List
import torch.nn as nn
from torch import Tensor

from layers.upsample import LateralUpsampleMerge
from layers.wrappers import conv1x1, conv3x3


class FPN(nn.Module):
    """
    Feature Pyramid Neural Network model with support for RetinaNet use-case.

    Notes:
      For standard FPN functionality pass empty array to `upsample_stages` args.

    For details see <paper>: https://arxiv.org/abs/1612.03144
    """

    def __init__(
        self,
        upsample_stages: List[int],
        downsample_stages: List[int],
        out_features=256,
    ):
        """
        Args:
          upsample_stages: list containing the input channels for each upsampling stage
            in ASC order (from lower level feature to higher level feature).
          downsample_stages: list containing the input channels for each downsampling stage
            in DESC order (from high level feature to lower level feature).
        """
        super().__init__()

        self.upsample_stage_names, self.upsample_stages = [], []

        total_stages = len(upsample_stages) + len(downsample_stages)
        up_start = total_stages - len(upsample_stages)
        ## pdb.set_trace()
        count = 0
        for _, in_channels in enumerate(upsample_stages):
            stage = conv3x3(in_channels, out_features, stride=2)
            stage_name = "upsample_fpn" + str(up_start + count)
            self.add_module(stage_name, stage)
            self.upsample_stages.append(stage)
            self.upsample_stage_names.append(stage_name)
            count += 1

        self.stage_names, self.stages, self.lat_stages = [], [], []
        count = 1
        for i, in_channels in enumerate(downsample_stages):
            if i == 0:
                lat_stage = conv1x1(in_channels, out_features)
            else:
                lat_stage = LateralUpsampleMerge(in_channels, out_features)
            conv_stage = conv3x3(out_features, out_features)
            stage_name = "fpn" + str(up_start - count)
            lat_name = "lat" + str(up_start - count)

            self.add_module(lat_name, lat_stage)
            self.add_module(stage_name, conv_stage)
            self.lat_stages.append(lat_stage)
            self.stages.append(conv_stage)
            self.stage_names.append(stage_name)
            count += 1

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: List[Tensor]):
        assert len(inputs) == len(self.stages)
        ## pdb.set_trace()

        outputs = {}

        inp_idxs = list(range(len(inputs) - 1, -1, -1))

        # Do upsampling.
        out = inputs[-1]
        upsample_idxs = list(range(len(self.upsample_stages) - 1, -1, -1))
        for i, name, stage in zip(
            upsample_idxs, self.upsample_stage_names, self.upsample_stages
        ):
            if i == len(self.upsample_stages) - 1:
                out = stage(out)
                outputs[name] = out
            else:
                out = stage(self.relu(out))
                outputs[name] = out

        # Do downsampling.
        count = 0
        for i, name, stage, lat_stage in zip(
            inp_idxs, self.stage_names, self.stages, self.lat_stages
        ):
            if count == 0:
                lat_out = lat_stage(inputs[i])
            else:
                lat_out = lat_stage(out, inputs[i])
            out = stage(lat_out)
            outputs[name] = out
            count += 1

        return outputs
