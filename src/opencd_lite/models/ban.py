"""BAN: Bi-Temporal Adapter Network over a frozen CLIP image tower.

Reference:
    K. Li et al., "A New Learning Paradigm for Foundation Model-based
    Remote-Sensing Change Detection," IEEE TGRS, vol. 62, 2024, doi:
    10.1109/TGRS.2024.3365825. https://arxiv.org/abs/2312.01163

Adapted from Open-CD (https://github.com/likyoo/open-cd, Apache-2.0).
Module attribute names intentionally match the Open-CD implementation so
that published Open-CD checkpoints load without key remapping
(``decode_head.*`` keys; the frozen CLIP tower lives under
``image_encoder.*`` in the detector).

The side-adapter branch is a MixVisionTransformer whose stages are fused
with projected CLIP features through cross-attention bridge layers; the
MLP decoder classifies the concatenated bi-temporal features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .changer_head import MixFFN
from .lightcdnet import LayerNorm2d
from .mit import MixVisionTransformer, _nchw_to_nlc, _nlc_to_nchw
from .modules import ConvModule
from .registry import register_head

__all__ = ["BitemporalAdapterHead"]


class CrossMultiheadAttention(nn.Module):
    """Cross-attention over NCHW maps (mmcv wrapper key layout, ``attn.*``)."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        kdim: int | None = None,
        vdim: int | None = None,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, kdim=kdim, vdim=vdim, bias=qkv_bias, batch_first=True
        )

    def forward(self, x_q: Tensor, x_kv: Tensor, identity: Tensor) -> Tensor:
        hw_shape = x_q.shape[-2:]
        x_q = _nchw_to_nlc(x_q)
        x_kv = _nchw_to_nlc(x_kv)
        out = self.attn(query=x_q, key=x_kv, value=x_kv, need_weights=False)[0]
        return identity + _nlc_to_nchw(out, (hw_shape[0], hw_shape[1]))


class BridgeLayer(nn.Module):
    """BAN bridging module: cross-attend side features to CLIP features.

    Args:
        num_heads: Attention heads.
        embed_dims: Channels of the side feature map.
        kdim: Key dimension (defaults to ``embed_dims``).
        vdim: Value dimension (defaults to ``embed_dims``).
        feedforward_channels: Hidden width of the (tiny) conv FFN;
            upstream leaves this at its default of 4.
    """

    def __init__(
        self,
        num_heads: int,
        embed_dims: int,
        kdim: int | None = None,
        vdim: int | None = None,
        feedforward_channels: int = 4,
    ) -> None:
        super().__init__()
        # Upstream normalizes with mmpretrain's channels-first LayerNorm
        # (default eps 1e-5).
        self.norm1 = LayerNorm2d(embed_dims, eps=1e-5)
        self.attn = CrossMultiheadAttention(embed_dims, num_heads, kdim=kdim, vdim=vdim)
        self.norm2 = LayerNorm2d(embed_dims, eps=1e-5)
        self.ffn = MixFFN(embed_dims=embed_dims, feedforward_channels=feedforward_channels)

    def forward(self, x: Tensor, x_kv: Tensor) -> Tensor:
        hw_shape = x.shape[-2:]
        x = self.attn(self.norm1(x), x_kv, identity=x)
        x = self.ffn(self.norm2(x), identity=x)
        return x + F.interpolate(x_kv, size=hw_shape, mode="bilinear", align_corners=False)


class BAN_MLPDecoder(nn.Module):
    """BAN's MLP mask decoder over the bi-temporal side features.

    Args:
        in_channels: Per-stage channels of the side-adapter features.
        channels: Common projection width.
        num_classes: Number of output classes.
        dropout_ratio: Dropout before the classifier.
        norm_cfg: mmseg-style norm config; only ``BN``/``SyncBN``.
        act_cfg: mmseg-style activation config; only ``ReLU``.
        align_corners: Bilinear upsampling alignment.
        interpolate_mode: Interpolation for the per-stage upsampling.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (32, 64, 160, 256),
        channels: int = 128,
        num_classes: int = 2,
        dropout_ratio: float = 0.1,
        norm_cfg: Mapping[str, Any] | None = None,
        act_cfg: Mapping[str, Any] | None = None,
        align_corners: bool = False,
        interpolate_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        if norm_cfg is not None and norm_cfg.get("type") not in ("BN", "SyncBN"):
            raise NotImplementedError(f"Unsupported norm_cfg {norm_cfg!r} (BN/SyncBN only)")
        if act_cfg is not None and act_cfg.get("type") != "ReLU":
            raise NotImplementedError(f"Unsupported act_cfg {act_cfg!r} (ReLU only)")

        self.in_channels = tuple(in_channels)
        self.channels = channels
        self.num_classes = num_classes
        self.out_channels = num_classes
        self.align_corners = align_corners
        self.interpolate_mode = interpolate_mode

        self.convs = nn.ModuleList(
            ConvModule(stage_channels, channels, 1, norm=True)
            for stage_channels in self.in_channels
        )
        self.fusion_conv = ConvModule(channels * len(self.in_channels), channels, 1, norm=True)
        self.discriminator = MixFFN(embed_dims=channels * 2, feedforward_channels=channels * 2)
        self.conv_seg = nn.Conv2d(channels * 2, self.out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None

    def base_forward(self, inputs: Sequence[Tensor]) -> Tensor:
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

    def forward(self, inputs1: Sequence[Tensor], inputs2: Sequence[Tensor]) -> Tensor:
        out1 = self.base_forward(inputs1)
        out2 = self.base_forward(inputs2)
        out = self.discriminator(torch.cat([out1, out2], dim=1))
        if self.dropout is not None:
            out = self.dropout(out)
        return self.conv_seg(out)


class BitemporalAdapterBranch(nn.Module):
    """The side-adapter encoder of BAN (MixVisionTransformer variant).

    Args:
        clip_channels: Channels of the CLIP tower features.
        fusion_index: Side-encoder stages fused with CLIP features.
        side_enc_cfg: Open-CD config dict of the side encoder; only
            ``type='mmseg.MixVisionTransformer'`` is supported.
    """

    def __init__(
        self,
        clip_channels: int | Sequence[int] = 768,
        fusion_index: Sequence[int] = (0, 1, 2),
        side_enc_cfg: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if side_enc_cfg is None:
            raise ValueError("side_enc_cfg is required")
        enc_cfg = dict(side_enc_cfg)
        enc_type = enc_cfg.pop("type")
        enc_cfg.pop("init_cfg", None)
        if enc_type != "mmseg.MixVisionTransformer":
            raise NotImplementedError(
                f"Side encoder {enc_type!r} is not supported yet (mmseg.MixVisionTransformer)"
            )
        self.side_encoder = MixVisionTransformer(**enc_cfg)
        self.fusion_index = tuple(fusion_index)

        side_enc_channels = [
            num * self.side_encoder.embed_dims for num in self.side_encoder.num_heads
        ]
        clip_channels_list = (
            [clip_channels] * len(side_enc_channels)
            if isinstance(clip_channels, int)
            else list(clip_channels)
        )
        self.conv_clips = nn.ModuleList(
            nn.Sequential(
                # mmseg's channels-first LayerNorm defaults to eps 1e-6.
                LayerNorm2d(clip_channels_list[i], eps=1e-6),
                ConvModule(clip_channels_list[i], side_enc_channels[i], 1, act=False),
            )
            for i in self.fusion_index
        )
        self.clip_attns = nn.ModuleList(
            BridgeLayer(num_heads=self.side_encoder.num_heads[i], embed_dims=side_enc_channels[i])
            for i in self.fusion_index
        )

    def fuse_clip(self, fused_index: int, x: Tensor, clip_feature: Tensor) -> Tensor:
        clip_fea = self.conv_clips[fused_index](clip_feature.contiguous())
        return self.clip_attns[fused_index](x, clip_fea)

    def forward(
        self, image: Tensor, clip_features: Sequence[Tensor | list[Tensor]]
    ) -> list[Tensor]:
        outs = []
        fused_index = 0
        x = image
        for index, layer in enumerate(self.side_encoder.layers):
            stage = cast(nn.ModuleList, layer)
            x, hw_shape = stage[0](x)
            for block in cast(nn.ModuleList, stage[1]):
                x = block(x, hw_shape)
            x = stage[2](x)
            x = _nlc_to_nchw(x, hw_shape)
            if index in self.fusion_index:
                clip_feature = clip_features[fused_index]
                if isinstance(clip_feature, list):
                    # [patch feature map, cls token]; only the map is used.
                    clip_feature = clip_feature[0]
                x = self.fuse_clip(fused_index, x, clip_feature)
                fused_index += 1
            outs.append(x)
        return outs


@register_head("BitemporalAdapterHead")
class BitemporalAdapterHead(nn.Module):
    """BAN decode head (Open-CD ``BitemporalAdapterHead``).

    Runs the shared side-adapter branch on each temporal image (fusing
    the frozen CLIP features of that image) and classifies the pair with
    the MLP mask decoder.

    Args:
        ban_cfg: Constructor arguments of
            :class:`BitemporalAdapterBranch`.
        ban_dec_cfg: Open-CD config dict of the mask decoder; only
            ``type='BAN_MLPDecoder'`` is supported.
    """

    def __init__(
        self,
        ban_cfg: Mapping[str, Any] | None = None,
        ban_dec_cfg: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if ban_cfg is None or ban_dec_cfg is None:
            raise ValueError("ban_cfg and ban_dec_cfg are required")
        dec_cfg = dict(ban_dec_cfg)
        dec_type = dec_cfg.pop("type")
        if dec_type != "BAN_MLPDecoder":
            raise NotImplementedError(
                f"Mask decoder {dec_type!r} is not supported yet (BAN_MLPDecoder)"
            )
        self.side_adapter_network = BitemporalAdapterBranch(**ban_cfg)
        self.mask_decoder = BAN_MLPDecoder(**dec_cfg)
        self.num_classes = self.mask_decoder.num_classes
        self.out_channels = self.mask_decoder.out_channels
        self.align_corners = self.mask_decoder.align_corners

    def forward(self, inputs: Sequence[Any]) -> Tensor:
        """Classify ``[img_from, img_to, clip_feats_from, clip_feats_to]``."""
        img_from, img_to, fm_feat_from, fm_feat_to = inputs
        mask_props_from = self.side_adapter_network(img_from, fm_feat_from)
        mask_props_to = self.side_adapter_network(img_to, fm_feat_to)
        return self.mask_decoder(mask_props_from, mask_props_to)
