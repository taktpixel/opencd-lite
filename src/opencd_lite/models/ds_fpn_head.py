"""DS-FPN decode head used by LightCDNet.

Adapted from Open-CD (https://github.com/likyoo/open-cd, Apache-2.0),
``opencd/models/decode_heads/ds_fpn_head.py``. Module attribute names
intentionally match upstream so that published checkpoints load without
key remapping (``decode_head.*`` keys).

The head consumes the *full* neck output pyramid: it drops the first
(early, deep-supervision-only) feature and sums per-scale projections of
the remaining FPN levels at the highest resolution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch.nn.functional as F
from torch import Tensor, nn

from .modules import ConvModule
from .registry import register_head

__all__ = ["DS_FPNHead"]


@register_head("DS_FPNHead")
class DS_FPNHead(nn.Module):
    """LightCDNet decode head (Open-CD ``DS_FPNHead``).

    Args:
        in_channels: Channels of each consumed pyramid level (after the
            early feature is dropped).
        channels: Common projection width (equal to the FPN width).
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

    #: The head receives the whole neck output tuple (it drops the early
    #: feature itself), instead of pre-selected ``in_index`` entries.
    takes_all_outputs = True

    def __init__(
        self,
        in_channels: Sequence[int] = (48, 48, 48, 48),
        channels: int = 48,
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

        self.in_channels = tuple(in_channels)
        self.channels = channels
        self.num_classes = num_classes
        self.out_channels = num_classes if out_channels is None else out_channels
        self.align_corners = align_corners

        self.scale_heads = nn.ModuleList(
            nn.Sequential(ConvModule(stage_channels, channels, 1, norm=True))
            for stage_channels in self.in_channels
        )
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)

    def forward(self, inputs: Sequence[Tensor]) -> Tensor:
        """Classify the neck pyramid (first, early feature is dropped)."""
        x = list(inputs[1:])
        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.in_channels)):
            output = output + F.interpolate(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        if self.dropout is not None:
            output = self.dropout(output)
        return self.conv_seg(output)
