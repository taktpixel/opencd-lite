"""BIT (Bitemporal Image Transformer) decode head.

Reference:
    H. Chen, Z. Qi and Z. Shi, "Remote Sensing Image Change Detection
    With Transformers," IEEE TGRS, vol. 60, 2022, doi:
    10.1109/TGRS.2021.3095166. Original implementation:
    https://github.com/justchenhao/BIT_CD (MIT license).

Adapted from Open-CD (https://github.com/likyoo/open-cd, Apache-2.0).
Module attribute names intentionally match the Open-CD implementation so
that published Open-CD checkpoints load without key remapping
(``decode_head.*`` keys).

The head receives the channel-concatenated bi-temporal feature map
produced by ``FeatureFusionNeck(policy='concat')``, splits it back into
the two temporal halves, compresses each into a few semantic tokens,
relates the tokens with a transformer encoder, projects them back onto
the feature maps with a transformer decoder, and classifies the absolute
feature difference.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .modules import ConvModule
from .registry import register_head

__all__ = ["BITHead"]


class CrossAttention(nn.Module):
    """Multi-head cross-attention with the BIT scaling convention.

    Note: upstream scales by ``in_dims ** -0.5`` (the block input width,
    not the per-head width); kept as-is for weight-exact behavior.
    """

    def __init__(
        self,
        in_dims: int,
        embed_dims: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        apply_softmax: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = in_dims**-0.5
        self.apply_softmax = apply_softmax

        self.to_q = nn.Linear(in_dims, embed_dims, bias=False)
        self.to_k = nn.Linear(in_dims, embed_dims, bias=False)
        self.to_v = nn.Linear(in_dims, embed_dims, bias=False)
        self.fc_out = nn.Sequential(nn.Linear(embed_dims, in_dims), nn.Dropout(dropout_rate))

    def forward(self, x: Tensor, ref: Tensor) -> Tensor:
        batch, length = x.shape[:2]
        heads = self.num_heads

        q = self.to_q(x).reshape(batch, length, heads, -1).permute(0, 2, 1, 3)
        k = self.to_k(ref).reshape(batch, ref.shape[1], heads, -1).permute(0, 2, 1, 3)
        v = self.to_v(ref).reshape(batch, ref.shape[1], heads, -1).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if self.apply_softmax:
            attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).flatten(2)
        return self.fc_out(out)


class FeedForward(nn.Sequential):
    """Two-layer MLP; plain ``nn.Sequential`` to match checkpoint keys."""

    def __init__(self, dim: int, hidden_dim: int, dropout_rate: float = 0.0) -> None:
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_rate),
        )


class TransformerEncoder(nn.Module):
    """Pre-norm self-attention block over the token sequence."""

    def __init__(
        self,
        in_dims: int,
        embed_dims: int,
        num_heads: int,
        drop_rate: float,
        apply_softmax: bool = True,
    ) -> None:
        super().__init__()
        self.attn = CrossAttention(
            in_dims, embed_dims, num_heads, dropout_rate=drop_rate, apply_softmax=apply_softmax
        )
        self.ff = FeedForward(in_dims, embed_dims, drop_rate)
        self.norm1 = nn.LayerNorm(in_dims)
        self.norm2 = nn.LayerNorm(in_dims)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn(self.norm1(x), self.norm1(x)) + x
        return self.ff(self.norm2(x)) + x


class TransformerDecoder(nn.Module):
    """Pre-norm cross-attention block projecting tokens onto pixels."""

    def __init__(
        self,
        in_dims: int,
        embed_dims: int,
        num_heads: int,
        drop_rate: float,
        apply_softmax: bool = True,
    ) -> None:
        super().__init__()
        self.attn = CrossAttention(
            in_dims, embed_dims, num_heads, dropout_rate=drop_rate, apply_softmax=apply_softmax
        )
        self.ff = FeedForward(in_dims, embed_dims, drop_rate)
        self.norm1 = nn.LayerNorm(in_dims)
        self.norm1_ = nn.LayerNorm(in_dims)
        self.norm2 = nn.LayerNorm(in_dims)

    def forward(self, x: Tensor, ref: Tensor) -> Tensor:
        x = self.attn(self.norm1(x), self.norm1_(ref)) + x
        return self.ff(self.norm2(x)) + x


@register_head("BITHead")
class BITHead(nn.Module):
    """BIT decode head classifying the fused bi-temporal feature map.

    Args:
        in_channels: Channels of one temporal half of the input (the
            head receives ``2 * in_channels`` concatenated channels).
        channels: Working width after ``pre_process``.
        embed_dims: Expanded width inside the attention blocks.
        enc_depth: Number of transformer encoder blocks.
        enc_with_pos: Add a learned position embedding to the tokens.
        dec_depth: Number of transformer decoder blocks.
        num_heads: Attention heads.
        drop_rate: Dropout inside the transformer blocks.
        pool_size: Token grid size when ``use_tokenizer`` is False.
        pool_mode: ``'max'`` or ``'avg'`` pooling for the same case.
        use_tokenizer: Compress features into semantic tokens with a
            learned attention map (otherwise pool to a fixed grid).
        token_len: Number of semantic tokens per temporal image.
        pre_upsample: Upsampling factor applied before ``pre_process``.
        upsample_size: Final upsampling factor of the feature map.
        num_classes: Number of output classes.
        out_channels: Output channels; defaults to ``num_classes``.
        threshold: Unused here (binarization happens in the inference
            wrapper); accepted for config compatibility.
        dropout_ratio: Dropout before the classifier.
        norm_cfg: mmseg-style norm config; only ``LN`` is supported.
        act_cfg: mmseg-style activation config; only ``ReLU`` is
            supported.
        align_corners: Bilinear upsampling alignment.
    """

    def __init__(
        self,
        in_channels: int = 256,
        channels: int = 32,
        embed_dims: int = 64,
        enc_depth: int = 1,
        enc_with_pos: bool = True,
        dec_depth: int = 8,
        num_heads: int = 8,
        drop_rate: float = 0.0,
        pool_size: int = 2,
        pool_mode: str = "max",
        use_tokenizer: bool = True,
        token_len: int = 4,
        pre_upsample: int = 2,
        upsample_size: int = 4,
        num_classes: int = 2,
        out_channels: int | None = None,
        threshold: float | None = None,
        dropout_ratio: float = 0.1,
        norm_cfg: Mapping[str, Any] | None = None,
        act_cfg: Mapping[str, Any] | None = None,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if norm_cfg is not None and norm_cfg.get("type") != "LN":
            raise NotImplementedError(f"Unsupported norm_cfg {norm_cfg!r} (LN only)")
        if act_cfg is not None and act_cfg.get("type") != "ReLU":
            raise NotImplementedError(f"Unsupported act_cfg {act_cfg!r} (ReLU only)")

        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.out_channels = num_classes if out_channels is None else out_channels
        self.align_corners = align_corners
        self.use_tokenizer = use_tokenizer
        self.enc_with_pos = enc_with_pos

        if use_tokenizer:
            self.token_len = token_len
            self.conv_att = ConvModule(channels, token_len, 1)
        else:
            self.pool_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = pool_size * pool_size

        if enc_with_pos:
            self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.token_len * 2, channels))

        self.pre_process = nn.Sequential(
            nn.Upsample(scale_factor=pre_upsample, mode="bilinear", align_corners=align_corners),
            ConvModule(in_channels, channels, 3, padding=1),
        )
        self.encoder = nn.ModuleList(
            TransformerEncoder(channels, embed_dims, num_heads, drop_rate=drop_rate)
            for _ in range(enc_depth)
        )
        self.decoder = nn.ModuleList(
            TransformerDecoder(channels, embed_dims, num_heads, drop_rate=drop_rate)
            for _ in range(dec_depth)
        )
        self.upsample = nn.Upsample(
            scale_factor=upsample_size, mode="bilinear", align_corners=align_corners
        )
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)

    def _forward_semantic_tokens(self, x: Tensor) -> Tensor:
        batch, channels = x.shape[:2]
        att_map = self.conv_att(x).reshape(batch, self.token_len, 1, -1)
        att_map = F.softmax(att_map, dim=-1)
        return (x.reshape(batch, 1, channels, -1) * att_map).sum(-1)

    def _forward_reshaped_tokens(self, x: Tensor) -> Tensor:
        if self.pool_mode == "max":
            x = F.adaptive_max_pool2d(x, (self.pool_size, self.pool_size))
        elif self.pool_mode == "avg":
            x = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        return x.permute(0, 2, 3, 1).flatten(1, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        """Classify a ``(B, 2 * in_channels, H, W)`` fused feature map."""
        x1, x2 = torch.chunk(inputs, 2, dim=1)
        x1 = self.pre_process(x1)
        x2 = self.pre_process(x2)

        if self.use_tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshaped_tokens(x1)
            token2 = self._forward_reshaped_tokens(x2)

        token = torch.cat([token1, token2], dim=1)
        if self.enc_with_pos:
            token = token + self.enc_pos_embedding
        for encoder in self.encoder:
            token = encoder(token)
        token1, token2 = torch.chunk(token, 2, dim=1)

        for decoder in self.decoder:
            batch, channels, height, width = x1.shape
            x1 = x1.permute(0, 2, 3, 1).flatten(1, 2)
            x2 = x2.permute(0, 2, 3, 1).flatten(1, 2)
            x1 = decoder(x1, token1)
            x2 = decoder(x2, token2)
            x1 = x1.transpose(1, 2).reshape(batch, channels, height, width)
            x2 = x2.transpose(1, 2).reshape(batch, channels, height, width)

        out = self.upsample(torch.abs(x1 - x2))
        if self.dropout is not None:
            out = self.dropout(out)
        return self.conv_seg(out)
