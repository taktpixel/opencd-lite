"""TinyCD v2 decode head.

Adapted from Open-CD (https://github.com/likyoo/open-cd, Apache-2.0),
``opencd/models/decode_heads/tiny_head.py``. Module attribute names
intentionally match upstream so that checkpoints load without key
remapping (``decode_head.*`` keys).

Sums per-scale projections of the FPN levels at the finest resolution
and, with ``priori_attn``, gates the result with the difference of the
early bi-temporal stem features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .modules import ConvModule
from .registry import register_head

__all__ = ["TinyHead"]


@register_head("TinyHead")
class TinyHead(nn.Module):
    """TinyCD v2 decode head (Open-CD ``TinyHead``).

    Args:
        in_channels: Channels of each consumed input. With
            ``priori_attn`` the first entry is the early stem feature
            used for the attention gate.
        feature_strides: Stride of each input feature map.
        priori_attn: Gate the fused features with the early-feature
            difference (Priori Guiding Connection).
        channels: Common projection width.
        num_classes: Number of output classes.
        out_channels: Output channels; defaults to ``num_classes``.
        threshold: Unused here (binarization happens in the inference
            wrapper); accepted for config compatibility.
        dropout_ratio: Dropout before the classifier.
        norm_cfg: mmseg-style norm config; only ``BN``/``SyncBN`` are
            supported.
        act_cfg: mmseg-style activation config; only ``ReLU`` is
            supported.
        align_corners: Bilinear upsampling alignment.
    """

    #: The head receives the whole neck output tuple (early feature plus
    #: FPN levels) and applies its own input handling.
    takes_all_outputs = True

    def __init__(
        self,
        in_channels: Sequence[int] = (32, 24, 24, 24, 24),
        feature_strides: Sequence[int] = (2, 2, 4, 8, 16),
        priori_attn: bool = False,
        channels: int = 24,
        num_classes: int = 2,
        out_channels: int | None = None,
        threshold: float | None = None,
        dropout_ratio: float = 0.1,
        norm_cfg: Mapping[str, Any] | None = None,
        act_cfg: Mapping[str, Any] | None = None,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if norm_cfg is not None and norm_cfg.get("type") not in ("BN", "SyncBN"):
            raise NotImplementedError(f"Unsupported norm_cfg {norm_cfg!r} (BN/SyncBN only)")
        if act_cfg is not None and act_cfg.get("type") != "ReLU":
            raise NotImplementedError(f"Unsupported act_cfg {act_cfg!r} (ReLU only)")
        if len(feature_strides) != len(in_channels):
            raise ValueError("feature_strides must match in_channels")
        if min(feature_strides) != feature_strides[0]:
            raise ValueError("The first feature stride must be the smallest")

        in_channels = tuple(in_channels)
        feature_strides = tuple(feature_strides)
        if priori_attn:
            attn_channels = in_channels[0]
            in_channels = in_channels[1:]
            feature_strides = feature_strides[1:]
        self.in_channels = in_channels
        self.feature_strides = feature_strides
        self.priori_attn = priori_attn
        self.channels = channels
        self.num_classes = num_classes
        self.out_channels = num_classes if out_channels is None else out_channels
        self.align_corners = align_corners

        self.scale_heads = nn.ModuleList(
            nn.Sequential(ConvModule(stage_channels, channels, 1, norm=True))
            for stage_channels in self.in_channels
        )
        if priori_attn:
            self.gen_diff_attn = ConvModule(attn_channels // 2, channels, 1, act=False)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)

    def forward(self, inputs: Sequence[Tensor]) -> Tensor:
        x = list(inputs)
        if self.priori_attn:
            early_x = x[0]
            x = x[1:]

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            output = output + F.interpolate(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

        if self.priori_attn:
            x1_, x2_ = torch.chunk(early_x, 2, dim=1)
            diff_x = self.gen_diff_attn(torch.abs(x1_ - x2_))
            if diff_x.shape != output.shape:
                output = F.interpolate(
                    output, diff_x.shape[2:], mode="bilinear", align_corners=self.align_corners
                )
            output = output * torch.sigmoid(diff_x) + output

        if self.dropout is not None:
            output = self.dropout(output)
        return self.conv_seg(output)
