"""ChangeStar with the FarSeg segmentation body.

References:
    Z. Zheng et al., "Change is Everywhere: Single-Temporal Supervised
    Object Change Detection in Remote Sensing Imagery," ICCV 2021
    (ChangeStar). https://arxiv.org/abs/2108.07002
    Z. Zheng et al., "Foreground-Aware Relation Network for Geospatial
    Object Segmentation in High Spatial Resolution Remote Sensing
    Imagery," CVPR 2020 (FarSeg). https://arxiv.org/abs/2011.09766

Adapted from Open-CD (https://github.com/likyoo/open-cd, Apache-2.0).
Module attribute names intentionally match the Open-CD implementation so
that published Open-CD checkpoints load without key remapping
(``neck.*`` / ``decode_head.*`` keys).

Layout: a siamese ResNet feeds :class:`FarSegFPN`, which builds an FPN
pyramid plus a global scene embedding per temporal image and
channel-concatenates the pairs. :class:`ChangeStarHead` splits them
back, runs the shared FarSeg segmentation body on each, and detects
change with the bidirectional :class:`ChangeMixin` module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .modules import ConvModule
from .necks import FeatureFusionNeck
from .registry import register_head
from .tiny_fpn import TinyFPN

__all__ = ["ChangeStarHead", "FarSegFPN", "FarSegHead"]


class FarSegFPN(TinyFPN):
    """FPN + global scene embedding, fused over the two temporal images.

    Args:
        policy: Fusion policy for the temporal pairs (``'concat'`` in
            the ChangeStar configs).
        in_channels: Backbone stage channels.
        out_channels: FPN width.
        num_outs: Number of FPN output scales.
    """

    def __init__(
        self,
        policy: str = "concat",
        in_channels: Sequence[int] = (64, 128, 256, 512),
        out_channels: int = 256,
        num_outs: int = 4,
    ) -> None:
        # mmseg's FPN with default configs matches TinyFPN's plain-conv
        # blocks (biased 1x1 laterals, biased 3x3 outputs, nearest
        # top-down upsampling), so the checkpoint key layout is shared.
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            custom_block="conv",
        )
        self.feature_fusion = FeatureFusionNeck(policy, out_indices=tuple(range(num_outs + 1)))

    def base_forward(self, inputs: Sequence[Tensor]) -> tuple[Tensor, ...]:
        fpn_feats = TinyFPN.forward(self, inputs)
        # Global scene embedding from the coarsest backbone feature.
        scene_embedding = F.adaptive_avg_pool2d(inputs[-1], 1)
        return (*fpn_feats, scene_embedding)

    def forward(  # type: ignore[override]
        self, x1: Sequence[Tensor], x2: Sequence[Tensor]
    ) -> tuple[Tensor, ...]:
        return self.feature_fusion(self.base_forward(x1), self.base_forward(x2))


class _FSRelation(nn.Module):
    """Foreground-scene relation module (FarSeg)."""

    def __init__(
        self,
        scene_embedding_channels: int,
        in_channels_list: Sequence[int],
        out_channels: int,
    ) -> None:
        super().__init__()
        self.scene_encoder = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(scene_embedding_channels, out_channels, 1),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 1),
            )
            for _ in in_channels_list
        )
        self.content_encoders = nn.ModuleList(
            nn.Sequential(ConvModule(channels, out_channels, 1, norm=True))
            for channels in in_channels_list
        )
        self.feature_reencoders = nn.ModuleList(
            nn.Sequential(ConvModule(channels, out_channels, 1, norm=True))
            for channels in in_channels_list
        )
        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature: Tensor, features: Sequence[Tensor]) -> list[Tensor]:
        content_feats = [
            encoder(feat) for encoder, feat in zip(self.content_encoders, features, strict=True)
        ]
        scene_feats = [encoder(scene_feature) for encoder in self.scene_encoder]
        relations = [
            self.normalizer((sf * cf).sum(dim=1, keepdim=True))
            for sf, cf in zip(scene_feats, content_feats, strict=True)
        ]
        p_feats = [
            encoder(feat) for encoder, feat in zip(self.feature_reencoders, features, strict=True)
        ]
        return [r * p for r, p in zip(relations, p_feats, strict=True)]


class _LightWeightDecoder(nn.Module):
    """FarSeg decoder: upsample every level to stride 4 and average."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_feature_output_strides: Sequence[int] = (4, 8, 16, 32),
        out_feature_output_stride: int = 4,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for in_feat_os in in_feature_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(
                math.log2(int(out_feature_output_stride))
            )
            num_layers = num_upsample if num_upsample != 0 else 1
            self.blocks.append(
                nn.Sequential(
                    *[
                        nn.Sequential(
                            ConvModule(
                                in_channels if idx == 0 else out_channels,
                                out_channels,
                                3,
                                padding=1,
                                norm=True,
                            ),
                            nn.UpsamplingBilinear2d(scale_factor=2)
                            if num_upsample != 0
                            else nn.Identity(),
                        )
                        for idx in range(num_layers)
                    ]
                )
            )

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        inner_feats = [block(feat) for block, feat in zip(self.blocks, features, strict=True)]
        return sum(inner_feats) / len(inner_feats)


class FarSegHead(nn.Module):
    """FarSeg segmentation body (Open-CD ``FarSegHead``).

    Args:
        in_channels: Channels of the FPN levels plus, last, the scene
            embedding.
        fsr_channels: Width of the foreground-scene relation module.
        channels: Decoder output width.
        num_classes: Number of output classes.
        out_channels: Output channels; defaults to ``num_classes``.
        dropout_ratio: Dropout before the classifier.
        norm_cfg: mmseg-style norm config; only ``BN``/``SyncBN`` are
            supported.
        act_cfg: mmseg-style activation config; only ``ReLU`` is
            supported.
        align_corners: Bilinear upsampling alignment.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (256, 256, 256, 256, 512),
        fsr_channels: int = 256,
        channels: int = 128,
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

        self._fsr = _FSRelation(self.in_channels[-1], self.in_channels[:-1], fsr_channels)
        self._decoder = _LightWeightDecoder(fsr_channels, channels)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg: nn.Module = nn.Conv2d(channels, self.out_channels, kernel_size=3)

    def forward(self, inputs: Sequence[Tensor]) -> Tensor:
        feats = list(inputs[:-1])
        scene_embedding = inputs[-1]
        refined = self._fsr(scene_embedding, feats)
        out = self._decoder(refined)
        if self.dropout is not None:
            out = self.dropout(out)
        return self.conv_seg(out)


class ChangeMixin(nn.Module):
    """Bidirectional change detector over concatenated temporal features.

    Args:
        in_channels: Sum of the two temporal feature widths.
        inner_channels: Width of the convolution blocks.
        num_convs: Number of convolution blocks.
    """

    def __init__(
        self, in_channels: int = 128 * 2, inner_channels: int = 16, num_convs: int = 4
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(in_channels, inner_channels, 3, 1, 1),
                nn.BatchNorm2d(inner_channels),
                nn.ReLU(True),
            )
        ]
        layers.extend(
            nn.Sequential(
                nn.Conv2d(inner_channels, inner_channels, 3, 1, 1),
                nn.BatchNorm2d(inner_channels),
                nn.ReLU(True),
            )
            for _ in range(num_convs - 1)
        )
        self.convs = nn.Sequential(*layers)

    def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
        c12 = self.convs(torch.cat([x1, x2], dim=1))
        c21 = self.convs(torch.cat([x2, x1], dim=1))
        return c12, c21


@register_head("ChangeStarHead")
class ChangeStarHead(nn.Module):
    """ChangeStar decode head (Open-CD ``ChangeStarHead``).

    Runs the shared FarSeg body on each temporal half of the fused
    pyramid and classifies the ChangeMixin output. At test time upstream
    selects one of the two directional predictions via
    ``inference_mode``; this forward reproduces exactly that selection.

    Args:
        inference_mode: ``'t1t2'``, ``'t2t1'`` or ``'mean'``.
        seg_head_cfg: Open-CD config dict of the segmentation body
            (only ``type='FarSegHead'`` is supported).
        changemixin_cfg: Constructor arguments of :class:`ChangeMixin`.
        in_channels: Accepted for config compatibility (upstream marks
            it as an unused placeholder).
        channels: Width of the ChangeMixin output (must equal its
            ``inner_channels``).
        num_classes: Number of classes.
        out_channels: Output channels (1 in the ChangeStar configs).
        threshold: Unused here (binarization happens in the inference
            wrapper); accepted for config compatibility.
        dropout_ratio: Dropout before the classifier.
        align_corners: Bilinear upsampling alignment.
    """

    #: The head consumes the full neck output tuple (FPN levels plus
    #: scene embedding) and splits the temporal pairs itself.
    takes_all_outputs = True

    def __init__(
        self,
        inference_mode: str = "t1t2",
        seg_head_cfg: Mapping[str, Any] | None = None,
        changemixin_cfg: Mapping[str, Any] | None = None,
        in_channels: Sequence[int] | None = None,
        channels: int = 96,
        num_classes: int = 2,
        out_channels: int | None = None,
        threshold: float | None = None,
        dropout_ratio: float = 0.1,
        norm_cfg: Mapping[str, Any] | None = None,
        act_cfg: Mapping[str, Any] | None = None,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if inference_mode not in ("t1t2", "t2t1", "mean"):
            raise ValueError(f"Invalid inference_mode: {inference_mode!r}")
        if seg_head_cfg is None or changemixin_cfg is None:
            raise ValueError("seg_head_cfg and changemixin_cfg are required")
        if channels != changemixin_cfg["inner_channels"]:
            raise ValueError("channels must equal changemixin_cfg['inner_channels']")

        self.inference_mode = inference_mode
        self.num_classes = num_classes
        self.out_channels = num_classes if out_channels is None else out_channels
        self.align_corners = align_corners

        seg_cfg = dict(seg_head_cfg)
        seg_type = seg_cfg.pop("type")
        if seg_type != "FarSegHead":
            raise NotImplementedError(f"seg_head type {seg_type!r} is not supported (FarSegHead)")
        seg_cfg.pop("in_index", None)
        seg_cfg.pop("loss_decode", None)
        # Upstream forces these on the inner head and replaces its
        # classifier with an identity (the ChangeMixin classifies).
        seg_cfg.update(num_classes=2, out_channels=1, dropout_ratio=0.0)
        self.seg_head = FarSegHead(**seg_cfg)
        self.seg_head.conv_seg = nn.Identity()

        self.changemixin = ChangeMixin(**changemixin_cfg)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)

    def _cls_seg(self, x: Tensor) -> Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        return self.conv_seg(x)

    def forward(self, inputs: Sequence[Tensor]) -> Tensor:
        inputs1 = []
        inputs2 = []
        for feature in inputs:
            f1, f2 = torch.chunk(feature, 2, dim=1)
            inputs1.append(f1)
            inputs2.append(f2)

        x1 = self.seg_head(inputs1)
        x2 = self.seg_head(inputs2)
        c12, c21 = self.changemixin(x1, x2)

        if self.inference_mode == "t1t2":
            return self._cls_seg(c12)
        if self.inference_mode == "t2t1":
            return self._cls_seg(c21)
        return (self._cls_seg(c12) + self._cls_seg(c21)) / 2.0
