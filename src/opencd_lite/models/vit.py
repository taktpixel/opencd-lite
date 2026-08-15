"""mmseg-compatible Vision Transformer (CLIP image tower) as plain torch.

Reimplements the subset of ``mmseg.models.backbones.VisionTransformer``
that the Open-CD BAN configs use: a CLIP-style pre-norm ViT with class
token, learned position embedding (bicubic resize for other input
sizes), QuickGELU activation and multi-stage feature output.

Module attribute names (``patch_embed.projection``, ``ln_pre``,
``layers.N.ln1/ln2``, ``layers.N.attn.attn`` — an
``nn.MultiheadAttention`` —, ``ffn.layers``) intentionally match mmseg
so that Open-CD checkpoints load without key remapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .mit import DropPath
from .registry import register_model

__all__ = ["VisionTransformer"]


class QuickGELU(nn.Module):
    """CLIP's GELU approximation: ``x * sigmoid(1.702 x)``."""

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(1.702 * x)


class _PatchEmbed(nn.Module):
    """Non-overlapping patch embedding (plain strided convolution)."""

    def __init__(
        self,
        in_channels: int,
        embed_dims: int,
        patch_size: int,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, embed_dims, patch_size, stride=patch_size, padding=padding, bias=bias
        )

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        x = self.projection(x)
        hw_shape = (x.shape[2], x.shape[3])
        return x.flatten(2).transpose(1, 2), hw_shape


class _Attention(nn.Module):
    """``nn.MultiheadAttention`` under the mmcv wrapper key layout (``attn.*``)."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, identity: Tensor) -> Tensor:
        out = self.attn(query=x, key=x, value=x, need_weights=False)[0]
        return identity + self.proj_drop(out)


class _FFN(nn.Module):
    """mmcv ``FFN`` (two linear layers) with the same submodule indices."""

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int,
        ffn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(embed_dims, feedforward_channels),
                QuickGELU(),
                nn.Dropout(ffn_drop),
            ),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(ffn_drop),
        )
        self.dropout_layer = DropPath(drop_path_rate)

    def forward(self, x: Tensor, identity: Tensor) -> Tensor:
        return identity + self.dropout_layer(self.layers(x))


class TransformerEncoderLayer(nn.Module):
    """Pre-norm ViT encoder block with mmseg attribute names."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        qkv_bias: bool = True,
        ln_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        # mmseg registers the norms as ln1/ln2 via build_norm_layer.
        self.ln1 = nn.LayerNorm(embed_dims, eps=ln_eps)
        self.attn = _Attention(
            embed_dims, num_heads, attn_drop=attn_drop_rate, proj_drop=drop_rate, qkv_bias=qkv_bias
        )
        self.ln2 = nn.LayerNorm(embed_dims, eps=ln_eps)
        # mmseg applies stochastic depth only on the FFN branch.
        self.ffn = _FFN(
            embed_dims, feedforward_channels, ffn_drop=drop_rate, drop_path_rate=drop_path_rate
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn(self.ln1(x), identity=x)
        return self.ffn(self.ln2(x), identity=x)


@register_model("mmseg.VisionTransformer")
class VisionTransformer(nn.Module):
    """CLIP-style Vision Transformer (mmseg ``VisionTransformer`` subset).

    Args:
        img_size: Position-embedding training resolution.
        patch_size: Patch size.
        patch_pad: Patch-embedding convolution padding.
        in_channels: Input image channels.
        embed_dims: Embedding dimension.
        num_layers: Number of encoder blocks.
        num_heads: Attention heads.
        mlp_ratio: FFN hidden width as a multiple of the embedding.
        out_origin: Also output the pre-transformer feature.
        out_indices: Blocks whose output is returned.
        qkv_bias: Enable bias for the attention projections.
        drop_rate: Dropout after projections/FFN.
        attn_drop_rate: Dropout on attention weights.
        drop_path_rate: Stochastic-depth rate.
        with_cls_token: Keep the class token in the sequence.
        output_cls_token: Return ``[feature_map, cls_token]`` pairs.
        patch_bias: Bias of the patch-embedding convolution.
        pre_norm: Apply a LayerNorm before the transformer (CLIP).
        final_norm: Apply a LayerNorm after the last block.
        norm_cfg: mmseg-style norm config; only ``LN`` is supported
            (``eps`` is honored).
        act_cfg: mmseg-style activation config; only ``mmseg.QuickGELU``
            and ``GELU`` are supported.
        norm_eval: Accepted for config compatibility (no BatchNorm here).
        interpolate_mode: Interpolation for position-embedding resizing.
        frozen_exclude: Parameter-name fragments excluded from freezing;
            everything else is frozen (upstream CLIP-tower behavior).
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = (224, 224),
        patch_size: int = 16,
        patch_pad: int = 0,
        in_channels: int = 3,
        embed_dims: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        out_origin: bool = False,
        out_indices: Sequence[int] = (2, 5, 8, 11),
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        with_cls_token: bool = True,
        output_cls_token: bool = False,
        patch_bias: bool = False,
        pre_norm: bool = False,
        final_norm: bool = False,
        norm_cfg: Mapping[str, Any] | None = None,
        act_cfg: Mapping[str, Any] | None = None,
        norm_eval: bool = False,
        interpolate_mode: str = "bicubic",
        frozen_exclude: Sequence[str] = ("all",),
    ) -> None:
        super().__init__()
        if norm_cfg is not None and norm_cfg.get("type") != "LN":
            raise NotImplementedError(f"Unsupported norm_cfg {norm_cfg!r} (LN only)")
        if act_cfg is not None and act_cfg.get("type") not in ("mmseg.QuickGELU", "GELU"):
            raise NotImplementedError(f"Unsupported act_cfg {act_cfg!r} (QuickGELU/GELU only)")
        if act_cfg is not None and act_cfg.get("type") == "GELU":
            raise NotImplementedError("Plain GELU ViTs are not needed by the ported configs")
        if output_cls_token and not with_cls_token:
            raise ValueError("output_cls_token requires with_cls_token")
        ln_eps = float(norm_cfg.get("eps", 1e-6)) if norm_cfg is not None else 1e-6

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = tuple(img_size)
        self.patch_size = patch_size
        self.interpolate_mode = interpolate_mode
        self.out_origin = out_origin
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.out_indices = tuple(out_indices)
        self.pre_norm = pre_norm
        self.final_norm = final_norm

        self.patch_embed = _PatchEmbed(
            in_channels, embed_dims, patch_size, padding=patch_pad, bias=patch_bias
        )
        num_patches = (self.img_size[0] // patch_size) * (self.img_size[1] // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        if pre_norm:
            self.ln_pre = nn.LayerNorm(embed_dims, eps=ln_eps)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=mlp_ratio * embed_dims,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                ln_eps=ln_eps,
            )
            for i in range(num_layers)
        )
        if final_norm:
            self.ln1 = nn.LayerNorm(embed_dims, eps=ln_eps)

        self._freeze(tuple(frozen_exclude))

    def _freeze(self, frozen_exclude: tuple[str, ...]) -> None:
        if "all" in frozen_exclude:
            return
        for name, param in self.named_parameters():
            if not any(fragment in name for fragment in frozen_exclude):
                param.requires_grad = False

    def _pos_embeding(self, x: Tensor, hw_shape: tuple[int, int]) -> Tensor:
        pos_embed: Tensor = self.pos_embed
        if x.shape[1] != pos_embed.shape[1]:
            pos_h = self.img_size[0] // self.patch_size
            pos_w = self.img_size[1] // self.patch_size
            pos_embed = self.resize_pos_embed(
                pos_embed, hw_shape, (pos_h, pos_w), self.interpolate_mode
            )
        return self.drop_after_pos(x + pos_embed)

    @staticmethod
    def resize_pos_embed(
        pos_embed: Tensor,
        input_shape: tuple[int, int],
        pos_shape: tuple[int, int],
        mode: str,
    ) -> Tensor:
        """Resize a ``[1, L, C]`` position embedding to a new patch grid."""
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0].unsqueeze(1)
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w) :]
        pos_embed_weight = pos_embed_weight.reshape(1, pos_h, pos_w, pos_embed.shape[2]).permute(
            0, 3, 1, 2
        )
        pos_embed_weight = F.interpolate(
            pos_embed_weight, size=input_shape, align_corners=False, mode=mode
        )
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        return torch.cat((cls_token_weight, pos_embed_weight), dim=1)

    def forward(self, inputs: Tensor) -> tuple[Tensor | list[Tensor], ...]:
        batch = inputs.shape[0]
        x, hw_shape = self.patch_embed(inputs)

        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape)

        if not self.with_cls_token:
            x = x[:, 1:]
        if self.pre_norm:
            x = self.ln_pre(x)

        outs: list[Tensor | list[Tensor]] = []
        if self.out_origin:
            outs.append(self._collect(x, batch, hw_shape))
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)
            if i in self.out_indices:
                outs.append(self._collect(x, batch, hw_shape))
        return tuple(outs)

    def _collect(self, x: Tensor, batch: int, hw_shape: tuple[int, int]) -> Tensor | list[Tensor]:
        out = x[:, 1:] if self.with_cls_token else x
        out = out.reshape(batch, hw_shape[0], hw_shape[1], -1).permute(0, 3, 1, 2).contiguous()
        if self.output_cls_token:
            return [out, x[:, 0]]
        return out
