"""Utility functions for reshaping Tensors in specific ways."""

from torch import Tensor


def flatten(x: Tensor):
    """Turn 3D tensor into 1D tensor."""
    return x.view(x.shape[0], -1)


def permute_to_N_HWA_K(tensor: Tensor, K: int):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (H x W x Ai), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # (N, HWA, K)
    return tensor