"""Preprocessing specification shared by training, inference and export.

Weight portability depends on preprocessing being reproduced exactly, so
the constants live here as *specification*, not configuration.

Open-CD preprocesses both CGNet and IFN identically
(``DualInputSegDataPreProcessor``):

* images are converted to **RGB** channel order,
* kept on the **0-255** scale and normalized with the ImageNet
  statistics below,
* padded on the bottom/right to a multiple of ``size_divisor`` (32)
  with zeros at test time.

Upstream reads files as BGR via OpenCV and sets ``bgr_to_rgb=True``;
reading images directly as RGB (e.g. with Pillow) is equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["PreprocessSpec", "IMAGENET_SPEC", "normalize_image", "pad_to_divisor"]


@dataclass(frozen=True)
class PreprocessSpec:
    """Normalization constants on the 0-255 RGB scale, plus padding rule."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    size_divisor: int = 32


#: ImageNet statistics used by every currently supported model.
IMAGENET_SPEC = PreprocessSpec(
    mean=(123.675, 116.28, 103.53),
    std=(58.395, 57.12, 57.375),
    size_divisor=32,
)


def normalize_image(image: np.ndarray | Tensor, spec: PreprocessSpec = IMAGENET_SPEC) -> Tensor:
    """Convert an RGB image to a normalized float tensor.

    Args:
        image: ``(H, W, 3)`` uint8/float array or tensor in RGB order on
            the 0-255 scale, or an already channel-first ``(3, H, W)``
            tensor on the 0-255 scale.
        spec: Normalization constants.

    Returns:
        Float32 tensor of shape ``(3, H, W)``.
    """
    if isinstance(image, np.ndarray):
        array = np.ascontiguousarray(image)
        if not array.flags.writeable:
            array = array.copy()
        tensor = torch.from_numpy(array)
    else:
        tensor = image
    tensor = tensor.float()
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3-dimensional image, got shape {tuple(tensor.shape)}")
    if tensor.shape[-1] == 3 and tensor.shape[0] != 3:
        tensor = tensor.permute(2, 0, 1)
    mean = tensor.new_tensor(spec.mean).view(3, 1, 1)
    std = tensor.new_tensor(spec.std).view(3, 1, 1)
    return (tensor - mean) / std


def pad_to_divisor(
    batch: Tensor, divisor: int, pad_value: float = 0.0
) -> tuple[Tensor, tuple[int, int]]:
    """Zero-pad a ``(B, C, H, W)`` batch on the bottom/right to a multiple of ``divisor``.

    Returns:
        The padded batch and the original ``(H, W)`` so callers can crop
        model outputs back afterwards.
    """
    height, width = batch.shape[-2:]
    pad_h = (divisor - height % divisor) % divisor
    pad_w = (divisor - width % divisor) % divisor
    if pad_h or pad_w:
        batch = F.pad(batch, (0, pad_w, 0, pad_h), value=pad_value)
    return batch, (height, width)
