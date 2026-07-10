"""CGNet: Change Guiding Network.

Reference:
    C. Han, C. Wu, H. Guo, M. Hu, J. Li and H. Chen,
    "Change Guiding Network: Incorporating Change Prior to Guide Change
    Detection in Remote Sensing Imagery," IEEE JSTARS, vol. 16,
    pp. 8395-8407, 2023, doi: 10.1109/JSTARS.2023.3310208.

Adapted from Open-CD (https://github.com/likyoo/open-cd, Apache-2.0).
Module attribute names intentionally match the Open-CD implementation so
that published Open-CD checkpoints load without key remapping.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import models

from .registry import register_model


class BasicConv2d(nn.Module):
    """Conv2d + BatchNorm + ReLU block."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


class ChangeGuideModule(nn.Module):
    """Self-attention module guided by a coarse change map."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.chanel_in = in_dim  # kept as-is (upstream spelling) for compatibility
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, guiding_map0: Tensor) -> Tensor:
        batch, channels, height, width = x.size()
        guiding_map0 = F.interpolate(
            guiding_map0, x.size()[2:], mode="bilinear", align_corners=True
        )
        guiding_map = torch.sigmoid(guiding_map0)

        query = self.query_conv(x) * (1 + guiding_map)
        proj_query = query.view(batch, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x) * (1 + guiding_map)
        proj_key = key.view(batch, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        value = self.value_conv(x) * (1 + guiding_map)
        proj_value = value.view(batch, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        return self.gamma * out + x


@register_model("CGNet")
class CGNet(nn.Module):
    """Change Guiding Network with a VGG-16 (BN) siamese encoder.

    The network is fully self-contained: in Open-CD terms it plays the
    "backbone" role while the decode/auxiliary heads are identity
    pass-throughs used only for loss computation.

    Args:
        pretrained: If True, initialize the VGG-16 encoder with ImageNet
            weights (downloads via torchvision on first use). Keep the
            default when training from scratch; set False when the full
            model is loaded from an Open-CD checkpoint afterwards.

    Forward inputs:
        x1, x2: Bi-temporal image tensors of shape ``(B, 3, H, W)``,
            preprocessed as described in :mod:`opencd_lite.transforms`.
            ``H`` and ``W`` must be divisible by 16.

    Forward returns:
        Tuple ``(change_map, final_map)`` of logits, each ``(B, 1, H, W)``.
        ``final_map`` (last element) is the primary prediction;
        ``change_map`` is the coarse guiding map used for deep supervision.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        vgg16_bn = models.vgg16_bn(weights=weights)
        self.inc = vgg16_bn.features[:5]  # 64 ch, full resolution
        self.down1 = vgg16_bn.features[5:12]  # 128 ch, 1/2
        self.down2 = vgg16_bn.features[12:22]  # 256 ch, 1/4
        self.down3 = vgg16_bn.features[22:32]  # 512 ch, 1/8
        self.down4 = vgg16_bn.features[32:42]  # 512 ch, 1/16

        self.conv_reduce_1 = BasicConv2d(128 * 2, 128, 3, 1, 1)
        self.conv_reduce_2 = BasicConv2d(256 * 2, 256, 3, 1, 1)
        self.conv_reduce_3 = BasicConv2d(512 * 2, 512, 3, 1, 1)
        self.conv_reduce_4 = BasicConv2d(512 * 2, 512, 3, 1, 1)

        # Unused in forward but present in upstream checkpoints; kept so
        # that ``load_state_dict(strict=True)`` succeeds on Open-CD weights.
        self.up_layer4 = BasicConv2d(512, 512, 3, 1, 1)
        self.up_layer3 = BasicConv2d(512, 512, 3, 1, 1)
        self.up_layer2 = BasicConv2d(256, 256, 3, 1, 1)

        self.decoder = nn.Sequential(BasicConv2d(512, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))
        self.decoder_final = nn.Sequential(BasicConv2d(128, 64, 3, 1, 1), nn.Conv2d(64, 1, 1))

        self.cgm_2 = ChangeGuideModule(256)
        self.cgm_3 = ChangeGuideModule(512)
        self.cgm_4 = ChangeGuideModule(512)

        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_module4 = BasicConv2d(1024, 512, 3, 1, 1)
        self.decoder_module3 = BasicConv2d(768, 256, 3, 1, 1)
        self.decoder_module2 = BasicConv2d(384, 128, 3, 1, 1)

    def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
        size = x1.size()[2:]
        layer1_pre = self.inc(x1)
        layer1_A = self.down1(layer1_pre)
        layer2_A = self.down2(layer1_A)
        layer3_A = self.down3(layer2_A)
        layer4_A = self.down4(layer3_A)

        layer1_pre = self.inc(x2)
        layer1_B = self.down1(layer1_pre)
        layer2_B = self.down2(layer1_B)
        layer3_B = self.down3(layer2_B)
        layer4_B = self.down4(layer3_B)

        layer1 = self.conv_reduce_1(torch.cat((layer1_B, layer1_A), dim=1))
        layer2 = self.conv_reduce_2(torch.cat((layer2_B, layer2_A), dim=1))
        layer3 = self.conv_reduce_3(torch.cat((layer3_B, layer3_A), dim=1))
        layer4 = self.conv_reduce_4(torch.cat((layer4_B, layer4_A), dim=1))

        feature_fuse = F.interpolate(layer4, layer1.size()[2:], mode="bilinear", align_corners=True)
        change_map = self.decoder(feature_fuse)

        layer4 = self.cgm_4(layer4, change_map)
        feature4 = self.decoder_module4(torch.cat([self.upsample2x(layer4), layer3], 1))
        layer3 = self.cgm_3(feature4, change_map)
        feature3 = self.decoder_module3(torch.cat([self.upsample2x(layer3), layer2], 1))
        layer2 = self.cgm_2(feature3, change_map)
        layer1 = self.decoder_module2(torch.cat([self.upsample2x(layer2), layer1], 1))

        change_map = F.interpolate(change_map, size, mode="bilinear", align_corners=True)
        final_map = self.decoder_final(layer1)
        final_map = F.interpolate(final_map, size, mode="bilinear", align_corners=True)

        return (change_map, final_map)
