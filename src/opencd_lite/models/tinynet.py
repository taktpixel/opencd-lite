"""TinyNet: the lightweight change-extractor backbone of TinyCD v2.

Adapted from Open-CD (https://github.com/likyoo/open-cd, Apache-2.0),
``opencd/models/backbones/tinynet.py``. Module attribute names
intentionally match the Open-CD implementation so that Open-CD
checkpoints load without key remapping.

MobileNetV2-style inverted-residual trunk over an early bi-temporal
fusion stem, with an asymmetric strip-convolution global attention after
each stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .lightcdnet import LayerNorm2d
from .modules import ConvModule
from .registry import register_model

__all__ = ["TinyBlock", "TinyNet"]


def make_divisible(value: float, divisor: int, min_value: int | None = None) -> int:
    """Round ``value`` to the nearest multiple of ``divisor`` (mmseg rule)."""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that rounding down does not go down by more than 10%.
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class AsymGlobalAttn(nn.Module):
    """Asymmetric strip-convolution global attention (after each stage).

    Args:
        dim: Feature channels.
        strip_kernel_size: Length of the 1D strip convolutions.
    """

    def __init__(self, dim: int, strip_kernel_size: int = 21) -> None:
        super().__init__()
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.global_ = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(
                dim,
                dim,
                (1, strip_kernel_size),
                padding=(0, (strip_kernel_size - 1) // 2),
                groups=dim,
            ),
            nn.Conv2d(
                dim,
                dim,
                (strip_kernel_size, 1),
                padding=((strip_kernel_size - 1) // 2, 0),
                groups=dim,
            ),
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.layer_scale = nn.Parameter(1e-6 * torch.ones(dim), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.global_(x)
        x = a * self.v(x)
        x = self.proj(x)
        x = self.norm(x)
        return self.layer_scale.unsqueeze(-1).unsqueeze(-1) * x + identity


class PriorAttention(nn.Module):
    """Difference-guided channel attention over the two temporal features."""

    def __init__(self, channels: int, num_paths: int = 2) -> None:
        super().__init__()
        self.num_paths = num_paths
        attn_channels = max(channels // 16, 8)
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(attn_channels)
        self.act = nn.ReLU(inplace=True)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
        attn = torch.abs(x1 - x2).mean((2, 3), keepdim=True)
        attn = self.act(self.bn(self.fc_reduce(attn)))
        attn = self.fc_select(attn)
        batch, channels, height, width = attn.shape
        attn1, attn2 = attn.reshape(batch, 2, channels // 2, height, width).transpose(0, 1)
        return x1 * torch.sigmoid(attn1) + x1, x2 * torch.sigmoid(attn2) + x2


class StemBlock(nn.Module):
    """Inverted-residual stem block with cross-temporal PriorAttention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")
        self.stride = stride
        self.use_res_connect = stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvModule(in_channels, hidden_dim, 1, norm=True, act="relu6"))
        layers.append(
            ConvModule(
                hidden_dim,
                hidden_dim,
                3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                norm=True,
                act="relu6",
            )
        )
        self.conv = nn.Sequential(*layers)
        self.interact = PriorAttention(channels=hidden_dim)
        self.post_conv = ConvModule(hidden_dim, out_channels, 1, norm=True, act=False)

    def forward(self, x: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x1, x2 = x
        identity_x1 = x1
        identity_x2 = x2
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        x1, x2 = self.interact(x1, x2)
        x1 = self.post_conv(x1)
        x2 = self.post_conv(x2)
        if self.use_res_connect:
            x1 = x1 + identity_x1
            x2 = x2 + identity_x2
        return x1, x2


class PriorFusion(nn.Module):
    """Early bi-temporal fusion stem producing the fused trunk input."""

    def __init__(self, channels: int, stack_nums: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            *[
                StemBlock(in_channels=channels, out_channels=channels, stride=1, expand_ratio=4)
                for _ in range(stack_nums)
            ]
        )
        self.pseudo_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2),
            LayerNorm2d(channels * 2, eps=1e-6),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 3, padding=1, groups=channels),
        )

    def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
        identity_x1 = x1
        identity_x2 = x2
        x1, x2 = self.stem((x1, x2))
        x1 = x1 + identity_x1
        x2 = x2 + identity_x2
        early_x = torch.cat([x1, x2], dim=1)
        x = self.pseudo_fusion(early_x)
        return early_x, x


class TinyBlock(nn.Module):
    """MobileNetV2 inverted-residual block (upstream ``TinyBlock``)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")
        self.stride = stride
        self.use_res_connect = stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvModule(in_channels, hidden_dim, 1, norm=True, act="relu6"))
        layers.extend(
            [
                ConvModule(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride=stride,
                    padding=dilation,
                    dilation=dilation,
                    groups=hidden_dim,
                    norm=True,
                    act="relu6",
                ),
                # Placeholder for the (unused) optional SE layer, keeping
                # upstream submodule indices.
                nn.Identity(),
                ConvModule(hidden_dim, out_channels, 1, norm=True, act=False),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


#: Per-stage (expand_ratio, channels, num_blocks) settings.
_ARCH_SETTINGS: dict[str, list[list[int]]] = {
    "S": [[4, 16, 2], [6, 24, 2], [6, 32, 3], [6, 48, 1]],
    "B": [[4, 16, 2], [6, 24, 2], [6, 32, 3], [6, 48, 1]],
    "L": [[4, 16, 2], [6, 24, 2], [6, 32, 6], [6, 48, 1]],
}


@register_model("TinyNet")
class TinyNet(nn.Module):
    """TinyNet backbone (Open-CD ``TinyNet``, TinyCD v2).

    Args:
        output_early_x: Prepend the pre-fusion concatenated stem feature
            to the outputs (consumed by TinyFPN's ``exist_early_x``).
        arch: Architecture variant: ``'S'``, ``'B'`` or ``'L'``.
        stem_stack_nums: Number of stacked stem blocks in the fusion.
        use_global: Whether to append :class:`AsymGlobalAttn` per stage.
        strip_kernel_size: Strip kernel length per stage.
        widen_factor: Channel width multiplier.
        strides: Stride of the first block of each stage.
        dilations: Dilation of each stage.
        out_indices: Stages whose output is returned.
        frozen_stages: Stages to freeze (-1 freezes nothing).
        conv_cfg: Accepted for config compatibility; must be ``None``.
        norm_cfg: mmseg-style norm config; only ``BN``/``SyncBN``.
        act_cfg: mmseg-style activation config; only ``ReLU6``.
        norm_eval: Keep BatchNorm layers in eval mode during training.
        with_cp: Accepted for config compatibility; must be ``False``.
        pretrained: Accepted for config compatibility; must be falsy
            (TinyNet has no published ImageNet weights).
        init_cfg: Accepted for config compatibility; ignored.
    """

    def __init__(
        self,
        output_early_x: bool = False,
        arch: str = "B",
        stem_stack_nums: int = 2,
        use_global: Sequence[bool] = (True, True, True, True),
        strip_kernel_size: Sequence[int] = (41, 31, 21, 11),
        widen_factor: float = 1.0,
        strides: Sequence[int] = (1, 2, 2, 2),
        dilations: Sequence[int] = (1, 1, 1, 1),
        out_indices: Sequence[int] = (0, 1, 2, 3),
        frozen_stages: int = -1,
        conv_cfg: Mapping[str, Any] | None = None,
        norm_cfg: Mapping[str, Any] | None = None,
        act_cfg: Mapping[str, Any] | None = None,
        norm_eval: bool = False,
        with_cp: bool = False,
        pretrained: str | bool | None = None,
        init_cfg: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if arch not in _ARCH_SETTINGS:
            raise ValueError(f"Unknown arch {arch!r} (S/B/L)")
        if conv_cfg is not None:
            raise NotImplementedError("conv_cfg is not supported (plain Conv2d only)")
        if norm_cfg is not None and norm_cfg.get("type") not in ("BN", "SyncBN"):
            raise NotImplementedError(f"Unsupported norm_cfg {norm_cfg!r} (BN/SyncBN only)")
        if act_cfg is not None and act_cfg.get("type") != "ReLU6":
            raise NotImplementedError(f"Unsupported act_cfg {act_cfg!r} (ReLU6 only)")
        if with_cp:
            raise NotImplementedError("with_cp is not supported")
        if pretrained:
            raise NotImplementedError("TinyNet has no supported pretrained weights")

        arch_settings = _ARCH_SETTINGS[arch]
        if not len(strides) == len(dilations) == len(arch_settings):
            raise ValueError("strides and dilations must have one entry per stage")
        self.out_indices = tuple(out_indices)
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.output_early_x = output_early_x

        self.in_channels = make_divisible(16 * widen_factor, 8)
        self.conv1 = ConvModule(3, self.in_channels, 3, stride=2, padding=1, norm=True, act="relu6")
        self.fusion_block = PriorFusion(self.in_channels, stem_stack_nums)

        self.layers: list[str] = []
        for i, (expand_ratio, channel, num_blocks) in enumerate(arch_settings):
            out_channels = make_divisible(channel * widen_factor, 8)
            layer = self._make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=strides[i],
                dilation=dilations[i],
                expand_ratio=expand_ratio,
                use_global=use_global[i],
                strip_kernel_size=strip_kernel_size[i],
            )
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, layer)
            self.layers.append(layer_name)

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int,
        dilation: int,
        expand_ratio: int,
        use_global: bool,
        strip_kernel_size: int,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        for i in range(num_blocks):
            layers.append(
                TinyBlock(
                    self.in_channels,
                    out_channels,
                    stride if i == 0 else 1,
                    expand_ratio=expand_ratio,
                    dilation=dilation if i == 0 else 1,
                )
            )
            self.in_channels = out_channels
        if use_global:
            layers.append(AsymGlobalAttn(out_channels, strip_kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor, ...]:
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)
        early_x, x = self.fusion_block(x1, x2)

        outs = [early_x] if self.output_early_x else []
        for i, layer_name in enumerate(self.layers):
            x = getattr(self, layer_name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f"layer{i}")
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True) -> TinyNet:
        """Set train mode, honoring ``frozen_stages`` and ``norm_eval``."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self
