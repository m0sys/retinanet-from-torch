"""
This module contains the implementation for a dynamic FPN model.

For details see <paper>: https://arxiv.org/abs/1612.03144
"""
from typing import List, Optional
import torch.nn as nn
from torch import Tensor

from layers.upsample import LateralUpsampleMerge
from layers.wrappers import conv1x1, conv3x3
from utils.weight_init import init_cnn


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
        out_feature_names: Optional[List[str]] = None,
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
        self._num_us = len(upsample_stages)
        self._num_ds = len(downsample_stages)
        self.out_feature_names = out_feature_names
        self.out_features = out_features

        u_feat_names = self._make_upsample_stages(upsample_stages)
        d_feat_names = self._make_downsample_stages(downsample_stages)

        if self.out_feature_names is None:
            self.out_feature_names = u_feat_names + d_feat_names

        self.relu = nn.ReLU(inplace=True)
        init_cnn(self)

    def _make_upsample_stages(self, upsample_stages):
        out_feat_names = []
        total_stages = self._num_us + self._num_ds
        up_start = total_stages - self._num_us
        self.upsample_stage_names, self.upsample_stages = [], []

        for i, in_channels in enumerate(upsample_stages):
            stage = conv3x3(in_channels, self.out_features, stride=2)
            stage_name = "upsample_fpn" + str(up_start + i)

            self.add_module(stage_name, stage)
            self.upsample_stages.append(stage)
            self.upsample_stage_names.append(stage_name)
            out_feat_names.append(stage_name)
        return out_feat_names

    def _make_downsample_stages(self, downsample_stages):
        out_feat_names = []
        total_stages = self._num_us + self._num_ds
        up_start = total_stages - self._num_us
        self.stage_names, self.stages, self.lat_stages = [], [], []

        for i, in_channels in enumerate(downsample_stages):
            if i == 0:
                lat_stage = conv1x1(in_channels, self.out_features)
            else:
                lat_stage = LateralUpsampleMerge(in_channels, self.out_features)

            conv_stage = conv3x3(self.out_features, self.out_features)
            stage_name = "fpn" + str(up_start - (i + 1))
            lat_name = "lat" + str(up_start - (i + 1))

            self.add_module(lat_name, lat_stage)
            self.add_module(stage_name, conv_stage)
            self.lat_stages.append(lat_stage)
            self.stages.append(conv_stage)
            self.stage_names.append(stage_name)
            out_feat_names.append(stage_name)
        return out_feat_names

    def forward(self, inputs: List[Tensor]):
        assert len(inputs) == len(self.stages)

        us_outputs = self._forward_upsampling_stages(inputs)
        ds_outputs = self._forward_downsampling_stages(inputs)

        return {**us_outputs, **ds_outputs}

    def _forward_upsampling_stages(self, inputs: List[Tensor]):
        """
        Run through upsampling part of the top-down path.

        This is what provides for the modification for the FPN
        model used in the RetinaNet paper.
        """
        outputs = {}

        out = inputs[-1]
        upsample_idxs = list(range(self._num_us))
        for i, name, stage in zip(
            upsample_idxs, self.upsample_stage_names, self.upsample_stages
        ):
            if i == 0:
                out = stage(out)
                outputs[name] = out
            else:
                out = stage(self.relu(out))
                if name in self.out_feature_names:
                    outputs[name] = out
        return outputs

    def _forward_downsampling_stages(self, inputs: List[Tensor]):
        """
        Run through downsampling part of the top-down path.

        This is the standard FPN forward implementation.
        """
        outputs = {}

        out = inputs[-1]
        rev_inp_idxs = list(range(len(inputs) - 1, -1, -1))
        for i, name, stage, lat_stage in zip(
            rev_inp_idxs, self.stage_names, self.stages, self.lat_stages
        ):
            if i == rev_inp_idxs[0]:
                lat_out = lat_stage(inputs[i])
            else:
                lat_out = lat_stage(out, inputs[i])
            out = stage(lat_out)
            if name in self.out_feature_names:
                outputs[name] = out
        return outputs


def retinanet_fpn_resnet(out_features=256, out_feat_names: Optional[List[str]] = None):
    return FPN(
        _RETINANET_FPN_RESNET["upsampling"],
        _RETINANET_FPN_RESNET["downsampling"],
        out_feat_names,
        out_features,
    )


_RETINANET_FPN_RESNET = {"upsampling": [2048, 256], "downsampling": [2048, 1024, 512]}
