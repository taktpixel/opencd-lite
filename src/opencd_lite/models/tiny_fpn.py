"""TinyFPN: small feature pyramid neck used by LightCDNet / TinyCD v2.

Adapted from Open-CD (https://github.com/likyoo/open-cd, Apache-2.0),
``opencd/models/necks/tiny_fpn.py``. Module attribute names
(``lateral_convs``, ``fpn_convs``) intentionally match upstream so that
published checkpoints load without key remapping (``neck.*`` keys).

Only the configuration space exercised by the Open-CD configs is
implemented: plain-conv blocks (``custom_block='conv'``) without extra
output levels. The TinyNet block variant belongs to the TinyCD v2 port.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn.functional as F
from torch import Tensor, nn

from .modules import ConvModule

__all__ = ["TinyFPN"]


class TinyFPN(nn.Module):
    """Feature Pyramid Network over an already-fused feature pyramid.

    Args:
        in_channels: Channels of each input scale (excluding the early
            feature when ``exist_early_x`` is set).
        out_channels: Common output width of every pyramid level.
        num_outs: Number of FPN output scales.
        custom_block: Block type for the per-level output convolutions;
            only ``'conv'`` (a plain 3x3 convolution) is supported.
        exist_early_x: The first input is an "early" high-resolution
            feature that is passed through unchanged and prepended to
            the outputs.
        early_x_for_fpn: Also feed that early feature into the pyramid
            (it then counts as the first ``in_channels`` entry).
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_outs: int,
        custom_block: str = "tinyblock",
        exist_early_x: bool = False,
        early_x_for_fpn: bool = False,
    ) -> None:
        super().__init__()
        if custom_block != "conv":
            raise NotImplementedError(
                f"custom_block {custom_block!r} is not supported yet ('conv' only)"
            )
        if num_outs < len(in_channels):
            raise ValueError("num_outs must cover every input scale")
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.exist_early_x = exist_early_x
        self.early_x_for_fpn = early_x_for_fpn

        # Upstream builds these with norm_cfg=None/act_cfg=None, i.e.
        # plain biased convolutions.
        self.lateral_convs = nn.ModuleList(
            ConvModule(channels, out_channels, 1, act=False) for channels in self.in_channels
        )
        self.fpn_convs = nn.ModuleList(
            ConvModule(out_channels, out_channels, 3, padding=1, act=False)
            for _ in self.in_channels
        )

    def forward(self, inputs: Sequence[Tensor]) -> tuple[Tensor, ...]:
        early_x: Tensor | None = None
        if self.exist_early_x:
            early_x = inputs[0]
            if not self.early_x_for_fpn:
                inputs = inputs[1:]
        if len(inputs) != len(self.in_channels):
            raise ValueError(f"Expected {len(self.in_channels)} input scales, got {len(inputs)}")

        laterals = [conv(x) for x, conv in zip(inputs, self.lateral_convs, strict=True)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="nearest"
            )

        outs = [conv(lateral) for lateral, conv in zip(laterals, self.fpn_convs, strict=True)]
        # Extra levels via max-pooling (upstream behavior without extra convs).
        while len(outs) < self.num_outs:
            outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        if early_x is not None:
            outs = [early_x, *outs]
        return tuple(outs)
