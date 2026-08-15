"""Interaction layers exchanging information between bi-temporal features.

Adapted from Open-CD (https://github.com/likyoo/open-cd, Apache-2.0),
``opencd/models/utils/interaction_layer.py``. All layers here are
parameter-free, so they contribute no checkpoint keys.

The exchange layers are rewritten functionally (``torch.where`` on a
broadcast mask instead of in-place masked assignment) for ONNX-export
friendliness; the results are identical because upstream applies the
same mask to every batch element.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

__all__ = ["ChannelExchange", "SpatialExchange", "TwoIdentity", "build_interaction_layer"]


class TwoIdentity(nn.Module):
    """No interaction; passes both features through unchanged."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
        return x1, x2


class ChannelExchange(nn.Module):
    """Exchange every ``1/p``-th channel between the two feature maps.

    Args:
        p: Fraction of channels to exchange (upstream keeps ``1/p`` as an
            integer period; channels whose index is a multiple of it are
            swapped).
    """

    def __init__(self, p: float = 1 / 2) -> None:
        super().__init__()
        if not 0 < p <= 1:
            raise ValueError(f"p must be in (0, 1], got {p}")
        self.p = int(1 / p)

    def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
        channels = x1.shape[1]
        exchange = (torch.arange(channels, device=x1.device) % self.p == 0).view(1, -1, 1, 1)
        return torch.where(exchange, x2, x1), torch.where(exchange, x1, x2)


class SpatialExchange(nn.Module):
    """Exchange every ``1/p``-th image column between the two feature maps.

    Args:
        p: Fraction of columns to exchange (same convention as
            :class:`ChannelExchange`).
    """

    def __init__(self, p: float = 1 / 2) -> None:
        super().__init__()
        if not 0 < p <= 1:
            raise ValueError(f"p must be in (0, 1], got {p}")
        self.p = int(1 / p)

    def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
        width = x1.shape[-1]
        exchange = (torch.arange(width, device=x1.device) % self.p == 0).view(1, 1, 1, -1)
        return torch.where(exchange, x2, x1), torch.where(exchange, x1, x2)


_INTERACTION_LAYERS: dict[str, type[nn.Module]] = {
    "TwoIdentity": TwoIdentity,
    "ChannelExchange": ChannelExchange,
    "SpatialExchange": SpatialExchange,
}


def build_interaction_layer(cfg: Mapping[str, Any] | None) -> nn.Module:
    """Build an interaction layer from an Open-CD ``interaction_cfg`` entry.

    ``None`` (no interaction configured for the stage) yields
    :class:`TwoIdentity`, matching upstream.
    """
    if cfg is None:
        return TwoIdentity()
    kwargs = dict(cfg)
    layer_type = kwargs.pop("type")
    try:
        layer_class = _INTERACTION_LAYERS[layer_type]
    except KeyError:
        supported = ", ".join(sorted(_INTERACTION_LAYERS))
        raise NotImplementedError(
            f"Interaction layer {layer_type!r} is not supported yet (supported: {supported})"
        ) from None
    return layer_class(**kwargs)
