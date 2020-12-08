import torch.nn as nn


def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
      module (torch.nn.Module): module to initialize.
    """
    # XavierFill in Caffe2 is the same as kaiming_uniform in PyTorch.
    nn.init.kaiming_uniform_(module.weight, a=1)  # set gain to 1.
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def c2_msra_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
      module (torch.nn.Module): module to initialize.
    """
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def init_c2msr_fill(m):
    """Initializes Conv layer weights using He Init with `fan_out`."""
    if getattr(m, "bias", None) is not None:
        nn.init.constant_(m.bias, 0)

    if isinstance(m, (nn.Conv2d)):
        c2_msra_fill(m)  # detectron init

    for l in m.children():
        init_c2msr_fill(l)


def init_cnn(m):
    """
    FastAI XResNet Init.

    see: https://github.com/fastai/fastai/blob/cad02a84e1adfd814bd97ff833a0d9661516308d/fastai/vision/models/xresnet.py#L16
    """
    if getattr(m, "bias", None) is not None:
        nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
    for l in m.children():
        init_cnn(l)
