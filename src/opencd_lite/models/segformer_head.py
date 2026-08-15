"""SegFormer all-MLP decode head.

Reference:
    E. Xie et al., "SegFormer: Simple and Efficient Design for Semantic
    Segmentation with Transformers," NeurIPS 2021.
    https://arxiv.org/abs/2105.15203

Reimplements ``mmseg.models.decode_heads.SegformerHead`` (from
mmsegmentation, Apache-2.0) without mmcv building blocks. Module
attribute names (``convs.N.conv/bn``, ``fusion_conv``, ``conv_seg``)
intentionally match mmseg so that Open-CD checkpoints load without key
remapping. ChangeFormer configures this head on the channel-concatenated
bi-temporal MiT features (doubled ``in_channels``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .modules import ConvModule
from .registry import register_head

__all__ = ["SegformerHead"]


@register_head("mmseg.SegformerHead")
class SegformerHead(nn.Module):
    """SegFormer decode head (mmseg ``SegformerHead``).

    Args:
        in_channels: Channels of each input feature map.
        channels: Common projection width.
        num_classes: Number of output classes.
        out_channels: Output channels; defaults to ``num_classes``.
        threshold: Unused here (binarization happens in the inference
            wrapper); accepted for config compatibility.
        dropout_ratio: Dropout before the classifier.
        interpolate_mode: Interpolation for the per-stage upsampling.
        norm_cfg: mmseg-style norm config; only ``BN``/``SyncBN`` are
            supported.
        act_cfg: mmseg-style activation config; only ``ReLU`` is
            supported.
        align_corners: Bilinear upsampling alignment.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (32, 64, 160, 256),
        channels: int = 256,
        num_classes: int = 2,
        out_channels: int | None = None,
        threshold: float | None = None,
        dropout_ratio: float = 0.1,
        interpolate_mode: str = "bilinear",
        norm_cfg: Mapping[str, Any] | None = None,
        act_cfg: Mapping[str, Any] | None = None,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if norm_cfg is not None and norm_cfg.get("type") not in ("BN", "SyncBN"):
            raise NotImplementedError(f"Unsupported norm_cfg {norm_cfg!r} (BN/SyncBN only)")
        if act_cfg is not None and act_cfg.get("type") != "ReLU":
            raise NotImplementedError(f"Unsupported act_cfg {act_cfg!r} (ReLU only)")

        self.in_channels = tuple(in_channels)
        self.channels = channels
        self.num_classes = num_classes
        self.out_channels = num_classes if out_channels is None else out_channels
        self.interpolate_mode = interpolate_mode
        self.align_corners = align_corners

        self.convs = nn.ModuleList(
            ConvModule(stage_channels, channels, 1, norm=True)
            for stage_channels in self.in_channels
        )
        self.fusion_conv = ConvModule(channels * len(self.in_channels), channels, 1, norm=True)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)

    def forward(self, inputs: Sequence[Tensor]) -> Tensor:
        """Fuse per-stage projections at the first stage's resolution."""
        outs = [
            F.interpolate(
                conv(x),
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners,
            )
            for x, conv in zip(inputs, self.convs, strict=True)
        ]
        out = self.fusion_conv(torch.cat(outs, dim=1))
        if self.dropout is not None:
            out = self.dropout(out)
        return self.conv_seg(out)
