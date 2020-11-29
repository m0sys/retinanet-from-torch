from abc import ABCMeta, abstractmethod
import torch.nn as nn

from layers.shape_spec import ShapeSpec


class Backbone(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for network backbones.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        """
        Maps input to a set of feature maps that are named using a dict.

        Returns:
          dict[str -> Tensor]: mapping feature map name (e.g. "res2") to feature map tensor.
        """
        pass

    @property
    def size_divisibility(self) -> int:
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return 0

    def output_shape(self):
        """
        Returns:
          dict[str -> ShapeSpec]
        """
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }
