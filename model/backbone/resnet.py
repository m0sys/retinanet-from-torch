from typing import Optional, List

from torchvision.models import (
    resnet50 as torch_resnet50,
    resnet101 as torch_resnet101,
    resnet152 as torch_resnet152,
)

from model.backbone.resnet_interface import ResNetInterface
from layers.resnet_blocks import (
    standard_bottleneck_block,
    StandardStem,
)


class ResNet(ResNetInterface):
    """
    Base class for creating all variants of ResNet while supporting FPN use-case.

    XResNet is derived from the following paper:
    "Bag of Tricks for Image Classification with Convolutional Neural Networks"

    """

    def __init__(
        self,
        layers: List[int],
        out_features: Optional[List[str]] = None,
        num_classes: Optional[int] = None,
        train_mode=False,
        use_dropout=False,
    ):
        stem = StandardStem(use_dropout=use_dropout)
        super().__init__(
            layers,
            stem,
            standard_bottleneck_block,
            out_features,
            num_classes,
            train_mode,
            use_dropout,
        )

    def load_pretrained_weights(self, pt_model):
        pt_list = list(pt_model.state_dict().items())
        count = 0
        model_dict = self.state_dict()
        for k, _ in model_dict.items():
            _, weights = pt_list[count]
            model_dict[k] = weights
            count += 1
        self.load_state_dict(model_dict)


def resnet50(
    out_features: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
    train_mode=False,
    use_dropout=False,
    pretrained=False,
):
    """Create a ResNet model 50 layers deep."""
    model = ResNet(_RESNET50_LAYERS, out_features, num_classes, train_mode, use_dropout)
    if pretrained:
        pt_model = torch_resnet50(pretrained=True)
        model.load_pretrained_weights(pt_model)
    return model


def resnet101(
    out_features: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
    train_mode=False,
    use_dropout=False,
    pretrained=False,
):
    """Create a ResNet model 101 layers deep."""
    model = ResNet(
        _RESNET101_LAYERS, out_features, num_classes, train_mode, use_dropout
    )
    if pretrained:
        pt_model = torch_resnet101(pretrained=True)
        model.load_pretrained_weights(pt_model)
    return model


def resnet152(
    out_features: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
    train_mode=False,
    use_dropout=False,
    pretrained=False,
):
    """Create a ResNet model 152 layers deep."""
    model = ResNet(
        _RESNET152_LAYERS, out_features, num_classes, train_mode, use_dropout
    )
    if pretrained:
        pt_model = torch_resnet152(pretrained=True)
        model.load_pretrained_weights(pt_model)
    return model


_RESNET34_LAYERS = [3, 4, 6, 3]
_RESNET50_LAYERS = [3, 4, 6, 3]
_RESNET101_LAYERS = [3, 4, 23, 3]
_RESNET152_LAYERS = [3, 8, 36, 3]
