"""Interaction ResNet: siamese ResNet with per-stage feature interactions.

Adapted from Open-CD (https://github.com/likyoo/open-cd, Apache-2.0),
``opencd/models/backbones/interaction_resnet.py``. The shared-weight
ResNet processes both temporal images; after every stage an interaction
layer (e.g. spatial/channel exchange) mixes the two feature maps, and
each returned stage output is the channel-concatenation of the pair.

The interaction layers hold no parameters, so the checkpoint key layout
is exactly that of the plain :class:`~opencd_lite.models.resnet.ResNet`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .interaction import build_interaction_layer
from .registry import register_model
from .resnet import ResNet

__all__ = ["IA_ResNet", "IA_ResNetV1c", "IA_ResNetV1d"]


@register_model("IA_ResNet")
class IA_ResNet(ResNet):
    """Interaction ResNet backbone (Open-CD ``IA_ResNet``).

    Args:
        interaction_cfg: One Open-CD interaction-layer config (or None)
            per stage, e.g. ``(None, {"type": "SpatialExchange", "p": 0.5},
            ...)``.
        **kwargs: Forwarded to :class:`~opencd_lite.models.resnet.ResNet`.
    """

    def __init__(
        self,
        interaction_cfg: Sequence[Mapping[str, Any] | None] = (None, None, None, None),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if len(interaction_cfg) != len(self.res_layers):
            raise ValueError(
                "interaction_cfg must have one entry per stage "
                f"({len(self.res_layers)}), got {len(interaction_cfg)}"
            )
        self.ccs = nn.ModuleList(build_interaction_layer(cfg) for cfg in interaction_cfg)

    def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor, ...]:  # type: ignore[override]
        x1 = self._stem_forward(x1)
        x2 = self._stem_forward(x2)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x1 = res_layer(x1)
            x2 = res_layer(x2)
            x1, x2 = self.ccs[i](x1, x2)
            if i in self.out_indices:
                outs.append(torch.cat([x1, x2], dim=1))
        return tuple(outs)


@register_model("IA_ResNetV1c")
class IA_ResNetV1c(IA_ResNet):
    """Interaction ResNet with a deep stem (three 3x3 convolutions)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(deep_stem=True, avg_down=False, **kwargs)


@register_model("IA_ResNetV1d")
class IA_ResNetV1d(IA_ResNet):
    """Deep-stem variant that also average-pools in downsample shortcuts."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(deep_stem=True, avg_down=True, **kwargs)
