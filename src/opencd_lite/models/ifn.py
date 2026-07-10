"""IFN: Deeply Supervised Image Fusion Network.

Reference:
    C. Zhang et al., "A deeply supervised image fusion network for change
    detection in high resolution bi-temporal remote sensing images,"
    ISPRS Journal of Photogrammetry and Remote Sensing, vol. 166, 2020.

Adapted from Open-CD (https://github.com/likyoo/open-cd, Apache-2.0),
which credits the original implementation at
https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images

Module attribute names intentionally match the Open-CD implementation so
that published Open-CD checkpoints load without key remapping.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: F401  (kept for parity with upstream imports)
from torch import Tensor, nn
from torchvision.models import VGG16_Weights, vgg16

from .registry import register_model


def make_norm(*args: int, **kwargs: object) -> nn.Module:
    """Create the normalization layer used throughout IFN (BatchNorm2d)."""
    return nn.BatchNorm2d(*args, **kwargs)  # type: ignore[arg-type]


class BasicConv(nn.Module):
    """Conv2d with optional padding, normalization and activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        pad_mode: str = "Zero",
        bias: bool | str = "auto",
        norm: bool | nn.Module = False,
        act: bool | nn.Module = False,
        **kwargs: object,
    ) -> None:
        super().__init__()
        seq: list[nn.Module] = []
        if kernel_size >= 2:
            seq.append(getattr(nn, pad_mode.capitalize() + "Pad2d")(kernel_size // 2))
        conv_bias = not norm if bias == "auto" else bool(bias)
        seq.append(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride=1,
                padding=0,
                bias=conv_bias,
                **kwargs,  # type: ignore[arg-type]
            )
        )
        if norm:
            seq.append(make_norm(out_ch) if norm is True else norm)
        if act:
            seq.append(nn.ReLU() if act is True else act)
        self.seq = nn.Sequential(*seq)

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


class Conv1x1(BasicConv):
    """1x1 convolution variant of :class:`BasicConv`."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        pad_mode: str = "Zero",
        bias: bool | str = "auto",
        norm: bool | nn.Module = False,
        act: bool | nn.Module = False,
        **kwargs: object,
    ) -> None:
        super().__init__(
            in_ch, out_ch, 1, pad_mode=pad_mode, bias=bias, norm=norm, act=act, **kwargs
        )


class ChannelAttention(nn.Module):
    """CBAM-style channel attention."""

    def __init__(self, in_ch: int, ratio: int = 8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = Conv1x1(in_ch, in_ch // ratio, bias=False, act=True)
        self.fc2 = Conv1x1(in_ch // ratio, in_ch, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = BasicConv(2, 1, kernel_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x))


class VGG16FeaturePicker(nn.Module):
    """Frozen VGG-16 feature extractor returning selected stage outputs."""

    def __init__(
        self, indices: tuple[int, ...] = (3, 8, 15, 22, 29), pretrained: bool = True
    ) -> None:
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        features = list(vgg16(weights=weights).features)[:30]
        self.features = nn.ModuleList(features).eval()
        self.indices = set(indices)

    def forward(self, x: Tensor) -> list[Tensor]:
        picked_feats = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.indices:
                picked_feats.append(x)
        return picked_feats


def conv2d_bn(in_ch: int, out_ch: int, with_dropout: bool = True) -> nn.Sequential:
    """Conv2d + PReLU + BatchNorm (+ optional Dropout) block."""
    lst: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.PReLU(),
        make_norm(out_ch),
    ]
    if with_dropout:
        lst.append(nn.Dropout(p=0.6))
    return nn.Sequential(*lst)


@register_model("IFN")
class IFN(nn.Module):
    """Deeply Supervised Image Fusion Network with a frozen VGG-16 encoder.

    The two encoder attributes reference the *same* module (shared
    weights), mirroring the upstream implementation; the encoder runs
    under ``torch.no_grad`` and is never trained.

    Args:
        use_dropout: Insert ``Dropout(p=0.6)`` in the decoder blocks.
        pretrained: If True, initialize the VGG-16 encoder with ImageNet
            weights (downloads via torchvision on first use). Set False
            when the full model is loaded from an Open-CD checkpoint
            afterwards.

    Forward inputs:
        t1, t2: Bi-temporal image tensors of shape ``(B, 3, H, W)``,
            preprocessed as described in :mod:`opencd_lite.transforms`.
            ``H`` and ``W`` must be divisible by 16.

    Forward returns:
        Tuple ``(out1, out2, out3, out4, out5)`` of logits at scales
        1/16, 1/8, 1/4, 1/2 and 1/1. ``out5`` (last element) is the
        primary full-resolution prediction; the rest are deep
        supervision outputs.
    """

    def __init__(self, use_dropout: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        # Shared frozen encoder, registered under both names as upstream.
        self.encoder1 = self.encoder2 = VGG16FeaturePicker(pretrained=pretrained)

        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa5 = SpatialAttention()

        self.ca1 = ChannelAttention(in_ch=1024)
        self.bn_ca1 = make_norm(1024)
        self.o1_conv1 = conv2d_bn(1024, 512, use_dropout)
        self.o1_conv2 = conv2d_bn(512, 512, use_dropout)
        self.bn_sa1 = make_norm(512)
        self.o1_conv3 = Conv1x1(512, 1)
        self.trans_conv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

        self.ca2 = ChannelAttention(in_ch=1536)
        self.bn_ca2 = make_norm(1536)
        self.o2_conv1 = conv2d_bn(1536, 512, use_dropout)
        self.o2_conv2 = conv2d_bn(512, 256, use_dropout)
        self.o2_conv3 = conv2d_bn(256, 256, use_dropout)
        self.bn_sa2 = make_norm(256)
        self.o2_conv4 = Conv1x1(256, 1)
        self.trans_conv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        self.ca3 = ChannelAttention(in_ch=768)
        self.o3_conv1 = conv2d_bn(768, 256, use_dropout)
        self.o3_conv2 = conv2d_bn(256, 128, use_dropout)
        self.o3_conv3 = conv2d_bn(128, 128, use_dropout)
        self.bn_sa3 = make_norm(128)
        self.o3_conv4 = Conv1x1(128, 1)
        self.trans_conv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

        self.ca4 = ChannelAttention(in_ch=384)
        self.o4_conv1 = conv2d_bn(384, 128, use_dropout)
        self.o4_conv2 = conv2d_bn(128, 64, use_dropout)
        self.o4_conv3 = conv2d_bn(64, 64, use_dropout)
        self.bn_sa4 = make_norm(64)
        self.o4_conv4 = Conv1x1(64, 1)
        self.trans_conv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.ca5 = ChannelAttention(in_ch=192)
        self.o5_conv1 = conv2d_bn(192, 64, use_dropout)
        self.o5_conv2 = conv2d_bn(64, 32, use_dropout)
        self.o5_conv3 = conv2d_bn(32, 16, use_dropout)
        self.bn_sa5 = make_norm(16)
        self.o5_conv4 = Conv1x1(16, 1)

    def forward(self, t1: Tensor, t2: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Extract bi-temporal features with the frozen encoder.
        with torch.no_grad():
            self.encoder1.eval()
            self.encoder2.eval()
            t1_feats = self.encoder1(t1)
            t2_feats = self.encoder2(t2)

        t1_f_l3, t1_f_l8, t1_f_l15, t1_f_l22, t1_f_l29 = t1_feats
        t2_f_l3, t2_f_l8, t2_f_l15, t2_f_l22, t2_f_l29 = t2_feats

        # Multi-level decoding with deep supervision outputs.
        x = torch.cat([t1_f_l29, t2_f_l29], dim=1)
        x = self.o1_conv1(x)
        x = self.o1_conv2(x)
        x = self.sa1(x) * x
        x = self.bn_sa1(x)
        out1 = self.o1_conv3(x)

        x = self.trans_conv1(x)
        x = torch.cat([x, t1_f_l22, t2_f_l22], dim=1)
        x = self.ca2(x) * x
        x = self.o2_conv1(x)
        x = self.o2_conv2(x)
        x = self.o2_conv3(x)
        x = self.sa2(x) * x
        x = self.bn_sa2(x)
        out2 = self.o2_conv4(x)

        x = self.trans_conv2(x)
        x = torch.cat([x, t1_f_l15, t2_f_l15], dim=1)
        x = self.ca3(x) * x
        x = self.o3_conv1(x)
        x = self.o3_conv2(x)
        x = self.o3_conv3(x)
        x = self.sa3(x) * x
        x = self.bn_sa3(x)
        out3 = self.o3_conv4(x)

        x = self.trans_conv3(x)
        x = torch.cat([x, t1_f_l8, t2_f_l8], dim=1)
        x = self.ca4(x) * x
        x = self.o4_conv1(x)
        x = self.o4_conv2(x)
        x = self.o4_conv3(x)
        x = self.sa4(x) * x
        x = self.bn_sa4(x)
        out4 = self.o4_conv4(x)

        x = self.trans_conv4(x)
        x = torch.cat([x, t1_f_l3, t2_f_l3], dim=1)
        x = self.ca5(x) * x
        x = self.o5_conv1(x)
        x = self.o5_conv2(x)
        x = self.o5_conv3(x)
        x = self.sa5(x) * x
        x = self.bn_sa5(x)
        out5 = self.o5_conv4(x)

        return (out1, out2, out3, out4, out5)
