"""MixVisionTransformer (SegFormer MiT) backbone as plain torch modules.

Reference:
    E. Xie et al., "SegFormer: Simple and Efficient Design for Semantic
    Segmentation with Transformers," NeurIPS 2021.
    https://arxiv.org/abs/2105.15203

Reimplements ``mmseg.models.backbones.MixVisionTransformer`` (from
mmsegmentation, Apache-2.0) without mmcv/mmengine building blocks.
Module attribute names (``layers.N.0.projection``,
``layers.N.1.M.attn.attn`` — an ``nn.MultiheadAttention`` —, ``ffn``,
per-stage ``norm``) intentionally match mmseg so that Open-CD
checkpoints trained with the mmseg backbone load without key remapping.

All LayerNorms use ``eps=1e-6`` (the mmseg MiT default), which matters
for numerical parity.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn

from .registry import register_model

__all__ = ["MixVisionTransformer"]

_LN_EPS = 1e-6


def _nlc_to_nchw(x: Tensor, hw_shape: tuple[int, int]) -> Tensor:
    height, width = hw_shape
    batch, _, channels = x.shape
    return x.transpose(1, 2).reshape(batch, channels, height, width).contiguous()


def _nchw_to_nlc(x: Tensor) -> Tensor:
    return x.flatten(2).transpose(1, 2).contiguous()


class DropPath(nn.Module):
    """Stochastic depth: randomly zero the residual branch per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep_prob * random.floor()


class PatchEmbed(nn.Module):
    """Overlapping patch embedding (strided convolution + LayerNorm)."""

    def __init__(
        self,
        in_channels: int,
        embed_dims: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, embed_dims, kernel_size, stride=stride, padding=padding
        )
        self.norm = nn.LayerNorm(embed_dims, eps=_LN_EPS)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        x = self.projection(x)
        hw_shape = (x.shape[2], x.shape[3])
        x = self.norm(_nchw_to_nlc(x))
        return x, hw_shape


class MixFFN(nn.Module):
    """SegFormer MixFFN: 1x1 conv MLP with a 3x3 depth-wise conv for position."""

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int,
        ffn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
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
        self.dropout_layer = DropPath(drop_path_rate)

    def forward(
        self, x: Tensor, hw_shape: tuple[int, int], identity: Tensor | None = None
    ) -> Tensor:
        out = _nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = _nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class EfficientMultiheadAttention(nn.Module):
    """Multi-head attention with optional spatial reduction of key/value.

    Wraps ``nn.MultiheadAttention`` under the attribute name ``attn`` to
    match the mmseg/mmcv checkpoint key layout (``attn.attn.in_proj_*``).
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        qkv_bias: bool = True,
        sr_ratio: int = 1,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True
        )
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = DropPath(drop_path_rate)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(embed_dims, embed_dims, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(embed_dims, eps=_LN_EPS)

    def forward(
        self, x: Tensor, hw_shape: tuple[int, int], identity: Tensor | None = None
    ) -> Tensor:
        x_q = x
        if self.sr_ratio > 1:
            x_kv = _nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = _nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x
        if identity is None:
            identity = x_q
        out = self.attn(query=x_q, key=x_kv, value=x_kv, need_weights=False)[0]
        return identity + self.dropout_layer(self.proj_drop(out))


class TransformerEncoderLayer(nn.Module):
    """One SegFormer encoder block: pre-norm attention + MixFFN."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        qkv_bias: bool = True,
        sr_ratio: int = 1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dims, eps=_LN_EPS)
        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            drop_path_rate=drop_path_rate,
            qkv_bias=qkv_bias,
            sr_ratio=sr_ratio,
        )
        self.norm2 = nn.LayerNorm(embed_dims, eps=_LN_EPS)
        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x: Tensor, hw_shape: tuple[int, int]) -> Tensor:
        x = self.attn(self.norm1(x), hw_shape, identity=x)
        return self.ffn(self.norm2(x), hw_shape, identity=x)


@register_model("mmseg.MixVisionTransformer")
class MixVisionTransformer(nn.Module):
    """SegFormer MiT backbone returning multi-stage NCHW feature maps.

    Args:
        in_channels: Number of input image channels.
        embed_dims: Base embedding dimension; stage ``i`` uses
            ``embed_dims * num_heads[i]``.
        num_stages: Number of stages.
        num_layers: Encoder blocks per stage.
        num_heads: Attention heads per stage.
        patch_sizes: Patch-embedding kernel size per stage.
        strides: Patch-embedding stride per stage.
        sr_ratios: Key/value spatial-reduction ratio per stage.
        out_indices: Stages whose output is returned.
        mlp_ratio: MixFFN hidden width as a multiple of the embedding.
        qkv_bias: Enable bias for the attention projections.
        drop_rate: Dropout after projections and inside MixFFN.
        attn_drop_rate: Dropout on attention weights.
        drop_path_rate: Maximum stochastic-depth rate (linearly scaled
            over the blocks).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 64,
        num_stages: int = 4,
        num_layers: Sequence[int] = (3, 4, 6, 3),
        num_heads: Sequence[int] = (1, 2, 4, 8),
        patch_sizes: Sequence[int] = (7, 3, 3, 3),
        strides: Sequence[int] = (4, 2, 2, 2),
        sr_ratios: Sequence[int] = (8, 4, 2, 1),
        out_indices: Sequence[int] = (0, 1, 2, 3),
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if not (
            num_stages
            == len(num_layers)
            == len(num_heads)
            == len(patch_sizes)
            == len(strides)
            == len(sr_ratios)
        ):
            raise ValueError("Per-stage settings must all have num_stages entries")
        self.num_stages = num_stages
        self.out_indices = tuple(out_indices)
        if max(self.out_indices) >= num_stages:
            raise ValueError(f"out_indices {self.out_indices} exceed num_stages {num_stages}")

        # Stochastic-depth decay rule over all blocks.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]

        cur = 0
        self.layers = nn.ModuleList()
        stage_in_channels = in_channels
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=stage_in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
            )
            blocks = nn.ModuleList(
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    sr_ratio=sr_ratios[i],
                )
                for idx in range(num_layer)
            )
            norm = nn.LayerNorm(embed_dims_i, eps=_LN_EPS)
            self.layers.append(nn.ModuleList([patch_embed, blocks, norm]))
            stage_in_channels = embed_dims_i
            cur += num_layer

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        outs = []
        for i, layer in enumerate(self.layers):
            stage = cast(nn.ModuleList, layer)
            x, hw_shape = stage[0](x)
            for block in cast(nn.ModuleList, stage[1]):
                x = block(x, hw_shape)
            x = stage[2](x)
            x = _nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
