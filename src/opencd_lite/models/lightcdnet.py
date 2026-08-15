"""LightCDNet: lightweight ShuffleNet-style change detection backbone.

Reference:
    Y. Xing et al., "LightCDNet: Lightweight Change Detection Network
    Based on VHR Images," IEEE GRSL, vol. 20, 2023, doi:
    10.1109/LGRS.2023.3304309.

Adapted from Open-CD (https://github.com/likyoo/open-cd, Apache-2.0).
Module attribute names intentionally match the Open-CD implementation so
that published Open-CD checkpoints load without key remapping.

The only compiled-extension dependency of the upstream backbone,
``mmcv.ops.CrissCrossAttention``, has been a pure-PyTorch module since
mmcv v1.3.13; :class:`CrissCrossAttention` reproduces that einsum-based
implementation (https://github.com/open-mmlab/mmcv/pull/1201), so no
compiled operator is needed.
"""

from __future__ import annotations

from itertools import accumulate

import torch
from torch import Tensor, nn

from .registry import register_model

__all__ = ["CrissCrossAttention", "LightCDNet"]


class LayerNorm2d(nn.Module):
    """Channels-first LayerNorm over NCHW feature maps (ConvNeXt style)."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class Scale(nn.Module):
    """Learnable scalar multiplier (mmcv ``Scale``; checkpoint key ``scale``)."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


def _neg_inf_diag(n: int, device: torch.device) -> Tensor:
    """[n, n] matrix with -inf on the diagonal (avoids double-counting).

    Built with masked_fill instead of ``torch.diag`` so the graph
    exports to ONNX.
    """
    index = torch.arange(n, device=device)
    diagonal = index.unsqueeze(0) == index.unsqueeze(1)
    return torch.zeros(n, n, device=device).masked_fill(diagonal, float("-inf"))


class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention (CCNet), pure-PyTorch mmcv implementation.

    Args:
        in_channels: Channels of the input feature map.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = Scale(0.0)
        self.in_channels = in_channels

    def forward(self, x: Tensor) -> Tensor:
        _, _, height, _ = x.size()
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        energy_h = torch.einsum("bchw,bciw->bwhi", query, key) + _neg_inf_diag(height, query.device)
        energy_h = energy_h.transpose(1, 2)
        energy_w = torch.einsum("bchw,bchj->bhwj", query, key)
        attn = torch.softmax(torch.cat([energy_h, energy_w], dim=-1), dim=-1)
        out = torch.einsum("bciw,bhwi->bchw", value, attn[..., :height])
        out = out + torch.einsum("bchj,bhwj->bchw", value, attn[..., height:])
        return (self.gamma(out) + x).contiguous()


class CCA(nn.Module):
    """Recurrent Criss-Cross Attention wrapper.

    Args:
        channels: Feature channels.
        recurrence: Number of criss-cross attention iterations.
    """

    def __init__(self, channels: int, recurrence: int = 2) -> None:
        super().__init__()
        self.recurrence = recurrence
        self.cca = CrissCrossAttention(channels)

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(self.recurrence):
            x = self.cca(x)
        return x


def channel_shuffle(x: Tensor, groups: int = 2) -> Tensor:
    """Interleave channels across groups (ShuffleNet)."""
    batch, channels, height, width = x.shape
    group_channels = channels // groups
    x = x.view(batch, groups, group_channels, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batch, -1, height, width)


class ShuffleBlock(nn.Module):
    """ShuffleNetV2 unit (channel split + depthwise conv + shuffle)."""

    def __init__(self, in_c: int, out_c: int, downsample: bool = False) -> None:
        super().__init__()
        self.downsample = downsample
        half_c = out_c // 2
        if downsample:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, 2, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True),
                nn.Conv2d(half_c, half_c, 3, 2, 1, groups=half_c, bias=False),
                nn.BatchNorm2d(half_c),
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True),
            )
        else:
            if in_c != out_c:
                raise ValueError("Non-downsampling ShuffleBlock requires in_c == out_c")
            self.branch2 = nn.Sequential(
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True),
                nn.Conv2d(half_c, half_c, 3, 1, 1, groups=half_c, bias=False),
                nn.BatchNorm2d(half_c),
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True),
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.downsample:
            out = torch.cat((self.branch1(x), self.branch2(x)), 1)
        else:
            channels = x.shape[1]
            c = channels // 2
            out = torch.cat((x[:, :c, :, :], self.branch2(x[:, c:, :, :])), 1)
        return channel_shuffle(out, 2)


class TimeAttention(nn.Module):
    """Channel attention re-weighting the two temporal features jointly."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        attn_channels = max(channels // 16, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels * 2, attn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(attn_channels),
            nn.ReLU(),
            nn.Conv2d(attn_channels, channels * 2, kernel_size=1, bias=False),
        )

    def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
        x = self.avg_pool(torch.cat((x1, x2), dim=1))
        y = self.mlp(x)
        batch, channels, height, width = y.size()
        x1_attn, x2_attn = y.reshape(batch, 2, channels // 2, height, width).transpose(0, 1)
        x1 = x1 * torch.sigmoid(x1_attn) + x1
        x2 = x2 * torch.sigmoid(x2_attn) + x2
        return x1, x2


class ShuffleFusion(nn.Module):
    """Early fusion of the two temporal stems (upstream ``shuffle_fusion``)."""

    def __init__(self, channels: int, block_num: int = 2) -> None:
        super().__init__()
        stages: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(channels, channels * 4, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels * 4),
                nn.ReLU(),
            )
        ]
        stages.extend(
            ShuffleBlock(channels * 4, channels * 4, downsample=False) for _ in range(block_num)
        )
        self.stages = nn.Sequential(*stages)
        self.single_conv = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.time_attn = TimeAttention(channels)
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward_single(self, x: Tensor) -> Tensor:
        return x + self.single_conv(self.stages(x))

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x1, x2 = self.time_attn(x1, x2)
        return self.final_conv(channel_shuffle(torch.cat((x1, x2), dim=1)))


#: Per-variant stage output channels and early-fusion depth.
_NET_SETTINGS: dict[str, tuple[list[int], int]] = {
    "small": ([24, 48, 96, 192], 4),
    "base": ([24, 116, 232, 464], 8),
    "large": ([24, 176, 352, 704], 16),
}


@register_model("LightCDNet")
class LightCDNet(nn.Module):
    """LightCDNet backbone (Open-CD ``LightCDNet``).

    Takes both temporal images (``DIEncoderDecoder`` layout), fuses them
    right after the stem and runs a single ShuffleNet-style trunk with
    criss-cross attention after each stage.

    Args:
        stage_repeat_num: Number of ShuffleBlocks per stage (e.g.
            ``[4, 8, 4]``).
        net_type: ``'small'``, ``'base'`` or ``'large'``.
    """

    def __init__(self, stage_repeat_num: list[int], net_type: str = "small") -> None:
        super().__init__()
        if net_type not in _NET_SETTINGS:
            raise ValueError(f"Unknown net_type {net_type!r} (small/base/large)")
        self.out_channels, self.block_num = _NET_SETTINGS[net_type]

        # Stage-output positions inside the flat `stages` sequence,
        # exactly as upstream computes them.
        index_list = list(stage_repeat_num)
        index_list[0] = index_list[0] - 1
        self.index_list = list(accumulate(index_list))

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.out_channels[0], 3, 2, 1, bias=False),
            LayerNorm2d(self.out_channels[0]),
            nn.GELU(),
        )
        self.fusion_conv = ShuffleFusion(self.out_channels[0], block_num=self.block_num)

        in_c = self.out_channels[0]
        stages: list[nn.Module] = []
        for stage_idx, repeat_num in enumerate(stage_repeat_num):
            out_c = self.out_channels[1 + stage_idx]
            for i in range(repeat_num):
                if i == 0:
                    stages.append(ShuffleBlock(in_c, out_c, downsample=True))
                else:
                    stages.append(ShuffleBlock(in_c, in_c, downsample=False))
                in_c = out_c
            stages.append(CCA(channels=out_c, recurrence=2))
        self.stages = nn.Sequential(*stages)

    def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor, ...]:
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)
        x = self.fusion_conv(x1, x2)
        outs = [x]
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            if i in self.index_list:
                outs.append(x)
        return tuple(outs)
