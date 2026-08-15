"""STANet decode head (spatial-temporal attention, metric output).

Reference:
    H. Chen and Z. Shi, "A Spatial-Temporal Attention-Based Method and a
    New Dataset for Remote Sensing Image Change Detection," Remote
    Sensing, vol. 12, no. 10, 2020, doi: 10.3390/rs12101662. Original
    implementation: https://github.com/justchenhao/STANet (BSD-2-Clause).

Adapted from Open-CD (https://github.com/likyoo/open-cd, Apache-2.0).
Module attribute names intentionally match the Open-CD implementation so
that published Open-CD checkpoints load without key remapping
(``decode_head.*`` keys).

Unlike the classification heads, STANet learns a feature embedding and
predicts the *Euclidean distance* between the two temporal embeddings.
Upstream binarizes at test time inside ``predict_by_feat`` by mapping
the distance map to +/-100 pseudo-logits before resizing; this head
reproduces exactly that in :meth:`STAHead.forward`, so the standard
sigmoid-threshold binarization (and sliding-window logit averaging) of
the inference wrapper yields upstream-identical masks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .modules import ConvModule
from .registry import register_head

__all__ = ["STAHead"]


class BAM(nn.Module):
    """Basic (global) self-attention module."""

    def __init__(self, in_dim: int, ds: int = 8) -> None:
        super().__init__()
        self.chanel_in = in_dim  # kept as-is (upstream spelling) for compatibility
        self.key_channel = in_dim // 8
        self.ds = ds
        self.pool = nn.AvgPool2d(ds)
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input: Tensor) -> Tensor:
        x = self.pool(input)
        batch, channels, width, height = x.size()
        proj_query = self.query_conv(x).view(batch, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch, -1, width * height)
        energy = torch.bmm(proj_query, proj_key) * self.key_channel**-0.5
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, width, height)
        out = F.interpolate(out, [width * self.ds, height * self.ds])
        return out + input


class _PAMBlock(nn.Module):
    """Pyramid attention block over one partition scale.

    Operates on the width-concatenated bi-temporal feature map
    ``(N, C, H, 2W)``, partitioning it into ``scale x scale`` regions and
    applying self-attention within each region (jointly over both
    temporal halves).
    """

    def __init__(
        self, in_channels: int, key_channels: int, value_channels: int, scale: int = 1, ds: int = 1
    ) -> None:
        super().__init__()
        self.scale = scale
        self.ds = ds
        self.pool = nn.AvgPool2d(ds)
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1),
            nn.BatchNorm2d(key_channels),
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1),
            nn.BatchNorm2d(key_channels),
        )
        self.f_value = nn.Conv2d(in_channels, value_channels, kernel_size=1)

    def forward(self, input: Tensor) -> Tensor:
        x = self.pool(input) if self.ds != 1 else input
        batch, _, h, double_w = x.shape
        w = double_w // 2

        local_y = []
        local_x = []
        step_h, step_w = h // self.scale, w // self.scale
        for i in range(self.scale):
            for j in range(self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x = h if i == self.scale - 1 else min(start_x + step_h, h)
                end_y = w if j == self.scale - 1 else min(start_y + step_w, w)
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]

        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)
        # Stack the two temporal halves along a trailing axis: B,C,H,W,2.
        value = torch.stack([value[:, :, :, :w], value[:, :, :, w:]], 4)
        query = torch.stack([query[:, :, :, :w], query[:, :, :, w:]], 4)
        key = torch.stack([key[:, :, :, :w], key[:, :, :, w:]], 4)

        local_block_cnt = 2 * self.scale * self.scale

        def attend(value_local: Tensor, query_local: Tensor, key_local: Tensor) -> Tensor:
            batch_local = value_local.size(0)
            h_local, w_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_local, self.value_channels, -1)
            query_local = query_local.contiguous().view(batch_local, self.key_channels, -1)
            query_local = query_local.permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_local, self.key_channels, -1)

            sim_map = torch.bmm(query_local, key_local) * self.key_channels**-0.5
            sim_map = F.softmax(sim_map, dim=-1)

            context_local = torch.bmm(value_local, sim_map.permute(0, 2, 1))
            return context_local.view(batch_local, self.value_channels, h_local, w_local, 2)

        # Batch all regions together for one attention call.
        v_locals = torch.cat(
            [
                value[:, :, local_x[i] : local_x[i + 1], local_y[i] : local_y[i + 1]]
                for i in range(0, local_block_cnt, 2)
            ],
            dim=0,
        )
        q_locals = torch.cat(
            [
                query[:, :, local_x[i] : local_x[i + 1], local_y[i] : local_y[i + 1]]
                for i in range(0, local_block_cnt, 2)
            ],
            dim=0,
        )
        k_locals = torch.cat(
            [
                key[:, :, local_x[i] : local_x[i + 1], local_y[i] : local_y[i + 1]]
                for i in range(0, local_block_cnt, 2)
            ],
            dim=0,
        )
        context_locals = attend(v_locals, q_locals, k_locals)

        context_rows = []
        for i in range(self.scale):
            row = [
                context_locals[batch * (j + i * self.scale) : batch * (j + i * self.scale) + batch]
                for j in range(self.scale)
            ]
            context_rows.append(torch.cat(row, dim=3))
        context = torch.cat(context_rows, dim=2)
        # Back to the width-concatenated layout.
        context = torch.cat([context[:, :, :, :, 0], context[:, :, :, :, 1]], dim=3)

        if self.ds != 1:
            context = F.interpolate(context, [h * self.ds, 2 * w * self.ds])
        return context


class PAMBlock(_PAMBlock):
    """`_PAMBlock` with the upstream default key/value widths."""

    def __init__(
        self,
        in_channels: int,
        key_channels: int | None = None,
        value_channels: int | None = None,
        scale: int = 1,
        ds: int = 1,
    ) -> None:
        if key_channels is None:
            key_channels = in_channels // 8
        if value_channels is None:
            value_channels = in_channels
        super().__init__(in_channels, key_channels, value_channels, scale, ds)


class PAM(nn.Module):
    """Pyramid attention module: parallel `_PAMBlock`s over several scales."""

    def __init__(
        self, in_channels: int, out_channels: int, sizes: Sequence[int] = (1,), ds: int = 1
    ) -> None:
        super().__init__()
        self.group = len(sizes)
        self.ds = ds
        self.value_channels = out_channels
        self.key_channels = out_channels // 8
        self.stages = nn.ModuleList(
            PAMBlock(in_channels, self.key_channels, self.value_channels, size, ds)
            for size in sizes
        )
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channels * self.group, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, feats: Tensor) -> Tensor:
        priors = [stage(feats) for stage in self.stages]
        return self.conv_bn(torch.cat(priors, dim=1))


def _weights_init(module: nn.Module) -> None:
    """Upstream STANet initialization (normal conv/BN init)."""
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.constant_(module.bias, 0)


class CDSA(nn.Module):
    """Change-detection self-attention over the concatenated image pair.

    Args:
        in_c: Input channels.
        ds: Downsampling factor inside the attention.
        mode: ``'BAM'``, ``'PAM'`` or ``'None'``.
    """

    def __init__(self, in_c: int, ds: int = 1, mode: str = "BAM") -> None:
        super().__init__()
        self.in_C = in_c
        self.ds = ds
        self.mode = mode
        if mode == "BAM":
            self.Self_Att: nn.Module = BAM(in_c, ds=ds)
        elif mode == "PAM":
            self.Self_Att = PAM(in_channels=in_c, out_channels=in_c, sizes=[1, 2, 4, 8], ds=ds)
        elif mode == "None":
            self.Self_Att = nn.Identity()
        else:
            raise NotImplementedError(f"Unsupported sa_mode {mode!r} (BAM/PAM/None)")
        self.apply(_weights_init)

    def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
        height = x1.shape[3]
        x = self.Self_Att(torch.cat((x1, x2), dim=3))
        return x[:, :, :, 0:height], x[:, :, :, height:]


@register_head("STAHead")
class STAHead(nn.Module):
    """STANet decode head (Open-CD ``STAHead``).

    Args:
        in_channels: Per-stage channels of one temporal half (the head
            receives ``2 * in_channels[i]`` concatenated channels per
            stage).
        channels: Common projection width of the FPN convolutions.
        sa_mode: Self-attention flavor: ``'BAM'``, ``'PAM'`` or
            ``'None'``.
        sa_in_channels: Embedding width fed into the attention module.
        sa_ds: Downsampling factor inside the attention module.
        distance_threshold: Distance above which a pixel is "changed".
        num_classes: Accepted for config compatibility (always 1).
        out_channels: Output channels; always 1 (the distance map).
        threshold: Unused here (binarization happens in the inference
            wrapper); accepted for config compatibility.
        norm_cfg: mmseg-style norm config; only ``BN``/``SyncBN`` are
            supported.
        act_cfg: mmseg-style activation config; only ``ReLU`` is
            supported.
        align_corners: Alignment of the final (wrapper-side) resize; the
            internal distance upsampling uses ``align_corners=True`` as
            upstream does.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (64, 128, 256, 512),
        channels: int = 96,
        sa_mode: str = "PAM",
        sa_in_channels: int = 256,
        sa_ds: int = 1,
        distance_threshold: float = 1.0,
        num_classes: int = 1,
        out_channels: int | None = None,
        threshold: float | None = None,
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
        self.num_classes = 1
        self.out_channels = 1
        self.distance_threshold = distance_threshold
        self.align_corners = align_corners

        self.fpn_convs = nn.ModuleList(
            ConvModule(stage_channels, channels, 1, norm=True)
            for stage_channels in self.in_channels
        )
        self.fpn_bottleneck = nn.Sequential(
            ConvModule(len(self.in_channels) * channels, sa_in_channels, 3, padding=1, norm=True),
            nn.Dropout(0.5),
            ConvModule(sa_in_channels, sa_in_channels, 3, padding=1, norm=True),
        )
        self.netA = CDSA(in_c=sa_in_channels, ds=sa_ds, mode=sa_mode)
        self.calc_dist = nn.PairwiseDistance(keepdim=True)
        # Upstream keeps a parameter-free identity classifier.
        self.conv_seg = nn.Identity()

    def base_forward(self, inputs: Sequence[Tensor]) -> Tensor:
        """Project each stage to ``channels`` and fuse at the 1/4 scale."""
        fpn_outs = [
            F.interpolate(
                conv(x),
                size=inputs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            for x, conv in zip(inputs, self.fpn_convs, strict=True)
        ]
        return self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))

    def forward_distance(self, inputs: Sequence[Tensor]) -> Tensor:
        """Raw embedding-distance map at the 1/4 scale (used by the BCL loss)."""
        inputs1 = []
        inputs2 = []
        for feature in inputs:
            f1, f2 = torch.chunk(feature, 2, dim=1)
            inputs1.append(f1)
            inputs2.append(f2)

        f1 = self.base_forward(inputs1)
        f2 = self.base_forward(inputs2)
        f1, f2 = self.netA(f1, f2)

        f1 = f1.permute(0, 2, 3, 1)
        f2 = f2.permute(0, 2, 3, 1)
        dist: Tensor = self.calc_dist(f1, f2).permute(0, 3, 1, 2)
        return F.interpolate(dist, size=inputs[0].shape[2:], mode="bilinear", align_corners=True)

    def forward(self, inputs: Sequence[Tensor]) -> Tensor:
        """Binarized +/-100 pseudo-logits (upstream test-time behavior)."""
        dist = self.forward_distance(inputs)
        return torch.where(
            dist > self.distance_threshold,
            torch.full_like(dist, 100.0),
            torch.full_like(dist, -100.0),
        )
