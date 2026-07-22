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

This module imports ``torch`` lazily: the constants
(:class:`PreprocessSpec`, :data:`IMAGENET_SPEC`) and the NumPy helpers
(:func:`normalize_image_numpy`, :func:`pad_to_divisor_numpy`) are usable
in a torch-free ONNX deployment, while :func:`normalize_image` and
:func:`pad_to_divisor` return ``torch`` tensors for the PyTorch path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "PreprocessSpec",
    "IMAGENET_SPEC",
    "normalize_image",
    "normalize_image_numpy",
    "pad_to_divisor",
    "pad_to_divisor_numpy",
]


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


def _to_chw_float(image: np.ndarray, spec: PreprocessSpec) -> np.ndarray:
    """Normalize an RGB image to a channel-first float32 array (no torch)."""
    array = np.asarray(image).astype(np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-dimensional image, got shape {array.shape}")
    if array.shape[-1] == 3 and array.shape[0] != 3:
        array = np.transpose(array, (2, 0, 1))
    mean = np.asarray(spec.mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(spec.std, dtype=np.float32).reshape(3, 1, 1)
    return (array - mean) / std


def normalize_image_numpy(image: np.ndarray, spec: PreprocessSpec = IMAGENET_SPEC) -> np.ndarray:
    """Convert an RGB image to a normalized ``(3, H, W)`` float32 array.

    The torch-free counterpart of :func:`normalize_image`, for ONNX
    deployment. Accepts ``(H, W, 3)`` or ``(3, H, W)`` uint8/float arrays
    on the 0-255 scale.
    """
    return np.ascontiguousarray(_to_chw_float(image, spec))


def pad_to_divisor_numpy(
    batch: np.ndarray, divisor: int, pad_value: float = 0.0
) -> tuple[np.ndarray, tuple[int, int]]:
    """Zero-pad an ``(N, C, H, W)`` array on the bottom/right to a multiple of ``divisor``.

    The torch-free counterpart of :func:`pad_to_divisor`.

    Returns:
        The padded array and the original ``(H, W)``.
    """
    height, width = batch.shape[-2:]
    pad_h = (divisor - height % divisor) % divisor
    pad_w = (divisor - width % divisor) % divisor
    if pad_h or pad_w:
        batch = np.pad(
            batch,
            ((0, 0), (0, 0), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=pad_value,
        )
    return batch, (height, width)


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
    import torch  # imported lazily so the constants stay torch-free

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
    import torch.nn.functional as F  # imported lazily so the constants stay torch-free

    height, width = batch.shape[-2:]
    pad_h = (divisor - height % divisor) % divisor
    pad_w = (divisor - width % divisor) % divisor
    if pad_h or pad_w:
        batch = F.pad(batch, (0, pad_w, 0, pad_h), value=pad_value)
    return batch, (height, width)
