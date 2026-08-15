"""mmseg-compatible ResNet backbones as plain torch modules.

Reimplements the subset of ``mmseg.models.backbones.ResNet`` that
Open-CD configs use (BasicBlock depths 18/34, ``style='pytorch'``,
optional deep stem), so published checkpoints trained with the mmseg
backbone load without key remapping.

Adapted from mmsegmentation (https://github.com/open-mmlab/mmsegmentation,
Apache-2.0). Module attribute names (``stem``, ``conv1``/``bn1``,
``layer<N>``, ``downsample``) intentionally match the mmseg
implementation so that Open-CD checkpoint keys map one-to-one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor, nn

from .registry import register_model

__all__ = ["BasicBlock", "ResNet", "ResNetV1c"]

#: Number of BasicBlocks per stage for the supported depths.
_ARCH_SETTINGS: dict[int, tuple[int, ...]] = {
    18: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
}
#: Open-CD configs request ``SyncBN`` (multi-GPU training); plain
#: ``BatchNorm2d`` is numerically identical at inference time.
_SUPPORTED_NORM_TYPES = ("BN", "SyncBN")


class BasicBlock(nn.Module):
    """ResNet BasicBlock (two 3x3 convolutions) with mmseg attribute names."""

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


@register_model("mmseg.ResNet")
class ResNet(nn.Module):
    """mmseg-style ResNet backbone returning multi-stage feature maps.

    Args:
        depth: Network depth; 18 and 34 (BasicBlock architectures) are
            supported.
        in_channels: Number of input image channels.
        stem_channels: Channels produced by the stem.
        base_channels: Channels of the first residual stage.
        num_stages: Number of residual stages to build (1-4).
        strides: Stride of the first block of each stage.
        dilations: Dilation of each stage.
        out_indices: Stages whose output is returned (0-based).
        style: Only ``'pytorch'`` (stride on the 3x3 convolution) is
            supported.
        deep_stem: Replace the 7x7 stem convolution with three 3x3
            convolutions (ResNetV1c).
        avg_down: Use AvgPool2d before the 1x1 convolution in downsample
            shortcuts (ResNetV1d).
        norm_cfg: mmseg-style norm config; only ``BN``/``SyncBN`` are
            supported (both map to :class:`torch.nn.BatchNorm2d`).
        norm_eval: Keep BatchNorm layers in eval mode (frozen running
            statistics) during training.
        contract_dilation: Halve the dilation of the first block in
            dilated stages (mmseg convention).
    """

    def __init__(
        self,
        depth: int = 18,
        in_channels: int = 3,
        stem_channels: int = 64,
        base_channels: int = 64,
        num_stages: int = 4,
        strides: Sequence[int] = (1, 2, 2, 2),
        dilations: Sequence[int] = (1, 1, 1, 1),
        out_indices: Sequence[int] = (0, 1, 2, 3),
        style: str = "pytorch",
        deep_stem: bool = False,
        avg_down: bool = False,
        norm_cfg: Mapping[str, Any] | None = None,
        norm_eval: bool = False,
        contract_dilation: bool = False,
    ) -> None:
        super().__init__()
        if depth not in _ARCH_SETTINGS:
            raise NotImplementedError(
                f"ResNet depth {depth} is not supported (supported: {sorted(_ARCH_SETTINGS)})"
            )
        if style != "pytorch":
            raise NotImplementedError("Only style='pytorch' is supported")
        if norm_cfg is not None and norm_cfg.get("type") not in _SUPPORTED_NORM_TYPES:
            raise NotImplementedError(f"Unsupported norm_cfg {norm_cfg!r} (BN/SyncBN only)")
        if not 1 <= num_stages <= 4:
            raise ValueError(f"num_stages must be in [1, 4], got {num_stages}")
        if len(strides) != num_stages or len(dilations) != num_stages:
            raise ValueError("strides and dilations must have num_stages entries")
        if any(index >= num_stages for index in out_indices):
            raise ValueError(f"out_indices {tuple(out_indices)} exceed num_stages {num_stages}")

        self.depth = depth
        self.deep_stem = deep_stem
        self.out_indices = tuple(out_indices)
        self.norm_eval = norm_eval

        if deep_stem:
            mid_channels = stem_channels // 2
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, stem_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, stem_channels, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(stem_channels)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_layers: list[str] = []
        inplanes = stem_channels
        for i, num_blocks in enumerate(_ARCH_SETTINGS[depth][:num_stages]):
            planes = base_channels * 2**i
            layer = self._make_layer(
                inplanes,
                planes,
                num_blocks,
                stride=strides[i],
                dilation=dilations[i],
                avg_down=avg_down,
                contract_dilation=contract_dilation,
            )
            inplanes = planes * BasicBlock.expansion
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, layer)
            self.res_layers.append(layer_name)

    @staticmethod
    def _make_layer(
        inplanes: int,
        planes: int,
        num_blocks: int,
        *,
        stride: int,
        dilation: int,
        avg_down: bool,
        contract_dilation: bool,
    ) -> nn.Sequential:
        out_channels = planes * BasicBlock.expansion
        downsample: nn.Module | None = None
        if stride != 1 or inplanes != out_channels:
            downsample_layers: list[nn.Module] = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample_layers.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False,
                    )
                )
            downsample_layers.extend(
                [
                    nn.Conv2d(inplanes, out_channels, 1, stride=conv_stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                ]
            )
            downsample = nn.Sequential(*downsample_layers)

        first_dilation = dilation // 2 if dilation > 1 and contract_dilation else dilation
        blocks = [
            BasicBlock(
                inplanes,
                planes,
                stride=stride,
                dilation=first_dilation,
                downsample=downsample,
            )
        ]
        blocks.extend(
            BasicBlock(out_channels, planes, dilation=dilation) for _ in range(1, num_blocks)
        )
        return nn.Sequential(*blocks)

    def _stem_forward(self, x: Tensor) -> Tensor:
        if self.deep_stem:
            return self.maxpool(self.stem(x))
        return self.maxpool(self.relu(self.bn1(self.conv1(x))))

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        x = self._stem_forward(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            x = getattr(self, layer_name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode: bool = True) -> ResNet:
        """Set train mode, keeping BatchNorm frozen when ``norm_eval`` is set."""
        super().train(mode)
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self


@register_model("mmseg.ResNetV1c")
class ResNetV1c(ResNet):
    """ResNet with a deep stem (three 3x3 convolutions instead of one 7x7).

    This is the variant Open-CD configs reference as ``mmseg.ResNetV1c``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(deep_stem=True, avg_down=False, **kwargs)
