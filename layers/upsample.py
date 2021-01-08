import torch.nn as nn
import torch.nn.functional as F

from layers.wrappers import conv1x1


class LateralUpsampleMerge(nn.Module):
    """Merge bottom-up path lateral connection with top-down upsampled path"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lat_conv = conv1x1(in_channels, out_channels)

    def forward(self, x, feature_map):
        lat_out = self.lat_conv(feature_map)
        return lat_out + F.interpolate(x, scale_factor=2.0, mode="nearest")