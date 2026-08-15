"""Changer decode head (flow dual-alignment fusion).

Reference:
    S. Fang, K. Li and Z. Li, "Changer: Feature Interaction Is What You
    Need for Change Detection," IEEE TGRS, vol. 61, 2023, doi:
    10.1109/TGRS.2023.3277496. https://arxiv.org/abs/2209.08290

Adapted from Open-CD (https://github.com/likyoo/open-cd, Apache-2.0).
Module attribute names intentionally match the Open-CD implementation so
that published Open-CD checkpoints load without key remapping
(``decode_head.*`` keys).

The head receives the multi-stage feature maps of an interaction
backbone (each stage the channel-concatenation of the two temporal
features), projects every stage to a common width, fuses the pyramid,
aligns the two temporal halves with a learned flow field (FDAF) and
classifies the fused difference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .modules import ConvModule
from .necks import FeatureFusionNeck
from .registry import register_head

__all__ = ["Changer"]


class FDAF(nn.Module):
    """Flow Dual-Alignment Fusion: warp each temporal feature onto the other.

    Args:
        in_channels: Channels of each temporal feature map.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        kernel_size = 5
        self.flow_make = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                in_channels * 2,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=True,
                groups=in_channels * 2,
            ),
            nn.InstanceNorm2d(in_channels * 2),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, 4, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x1: Tensor, x2: Tensor, fusion_policy: str | None = None) -> Tensor:
        flow = self.flow_make(torch.cat([x1, x2], dim=1))
        f1, f2 = torch.chunk(flow, 2, dim=1)
        x1_feat = self.warp(x1, f1) - x2
        x2_feat = self.warp(x2, f2) - x1
        if fusion_policy is None:
            raise ValueError("fusion_policy is required")
        return FeatureFusionNeck.fusion(x1_feat, x2_feat, fusion_policy)

    @staticmethod
    def warp(x: Tensor, flow: Tensor) -> Tensor:
        """Bilinearly sample ``x`` at positions displaced by ``flow`` (pixels)."""
        n, _, h, w = x.size()
        norm = torch.tensor([[[[w, h]]]], dtype=x.dtype, device=x.device)
        col = torch.linspace(-1.0, 1.0, h, dtype=x.dtype, device=x.device)
        row = torch.linspace(-1.0, 1.0, w, dtype=x.dtype, device=x.device)
        grid = torch.cat(
            [row.repeat(h, 1).unsqueeze(2), col.view(-1, 1).repeat(1, w).unsqueeze(2)], dim=2
        )
        grid = grid.repeat(n, 1, 1, 1) + flow.permute(0, 2, 3, 1) / norm
        return F.grid_sample(x, grid, align_corners=True)


class MixFFN(nn.Module):
    """SegFormer MixFFN used as the projection head of Changer.

    A 1x1 conv expansion, a 3x3 depth-wise conv providing positional
    information, GELU, and a 1x1 conv projection, with a residual
    shortcut. Submodule indices inside ``layers`` match upstream.
    """

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int,
        ffn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.activate = nn.GELU()
        drop = nn.Dropout(ffn_drop)
        self.layers = nn.Sequential(
            nn.Conv2d(embed_dims, feedforward_channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(
                feedforward_channels,
                feedforward_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=feedforward_channels,
            ),
            self.activate,
            drop,
            nn.Conv2d(feedforward_channels, embed_dims, kernel_size=1, stride=1, bias=True),
            drop,
        )
        # Upstream configures DropPath with drop_prob=0, i.e. an identity.
        self.dropout_layer = nn.Identity()

    def forward(self, x: Tensor, identity: Tensor | None = None) -> Tensor:
        out = self.layers(x)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


@register_head("Changer")
class Changer(nn.Module):
    """Changer decode head (Open-CD ``Changer``).

    Args:
        in_channels: Per-stage channels of one temporal half (the head
            receives ``2 * in_channels[i]`` concatenated channels per
            stage).
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
        in_channels: Sequence[int] = (64, 128, 256, 512),
        channels: int = 128,
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
        self.fusion_conv = ConvModule(channels * len(self.in_channels), channels // 2, 1, norm=True)
        self.neck_layer = FDAF(in_channels=channels // 2)
        self.discriminator = MixFFN(embed_dims=channels, feedforward_channels=channels)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)

    def base_forward(self, inputs: Sequence[Tensor]) -> Tensor:
        """Project each stage to ``channels`` and fuse at the 1/4 scale."""
        outs = [
            F.interpolate(
                conv(x),
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners,
            )
            for x, conv in zip(inputs, self.convs, strict=True)
        ]
        return self.fusion_conv(torch.cat(outs, dim=1))

    def forward(self, inputs: Sequence[Tensor]) -> Tensor:
        """Classify multi-stage concatenated bi-temporal feature maps."""
        inputs1 = []
        inputs2 = []
        for feature in inputs:
            f1, f2 = torch.chunk(feature, 2, dim=1)
            inputs1.append(f1)
            inputs2.append(f2)

        out1 = self.base_forward(inputs1)
        out2 = self.base_forward(inputs2)
        out = self.neck_layer(out1, out2, "concat")
        out = self.discriminator(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return self.conv_seg(out)
