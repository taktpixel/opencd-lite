"""FC-EF / FC-Siam-diff / FC-Siam-conc: fully convolutional siamese networks.

Reference:
    Daudt, R. C., Le Saux, B., & Boulch, A.
    "Fully convolutional siamese networks for change detection,"
    2018 25th IEEE International Conference on Image Processing (ICIP),
    pp. 4063-4067.

Adapted from Open-CD (https://github.com/likyoo/open-cd, Apache-2.0),
which in turn borrows from:
https://github.com/rcdaudt/fully_convolutional_change_detection
https://github.com/Bobholamovic/CDLab
https://github.com/likyoo/Siam-NestedUNet

The three networks share the U-Net topology and differ only in how the
bi-temporal branches are fused (early concat / feature difference /
feature concat). They are kept as separate flat classes — mirroring
upstream — because module attribute names must match the Open-CD
checkpoints key-for-key.

Unlike CGNet/IFN, these models output feature maps: the classification
happens in a separate parametric decode head
(:class:`~opencd_lite.models.heads.ConvSegHead`, built from the config).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.padding import ReplicationPad2d

from .registry import register_model


@register_model("FC_EF")
class FC_EF(nn.Module):
    """FC-EF: early-fusion U-Net over channel-concatenated image pairs.

    Args:
        in_channels: Channels of the fused input (6 for RGB pairs).
        base_channel: Width of the first stage.

    Forward returns a 1-tuple with a ``(B, base_channel, H, W)`` feature
    map at input resolution.
    """

    def __init__(self, in_channels: int, base_channel: int = 16) -> None:
        super().__init__()

        filters = [
            base_channel,
            base_channel * 2,
            base_channel * 4,
            base_channel * 8,
            base_channel * 16,
        ]

        self.conv11 = nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(filters[0])
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(filters[0])
        self.do12 = nn.Dropout2d(p=0.2)

        self.conv21 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(filters[1])
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(filters[1])
        self.do22 = nn.Dropout2d(p=0.2)

        self.conv31 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(filters[2])
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(filters[2])
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(filters[2])
        self.do33 = nn.Dropout2d(p=0.2)

        self.conv41 = nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(filters[3])
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(filters[3])
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(filters[3])
        self.do43 = nn.Dropout2d(p=0.2)

        self.upconv4 = nn.ConvTranspose2d(
            filters[3], filters[3], kernel_size=3, padding=1, stride=2, output_padding=1
        )

        self.conv43d = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(filters[3])
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(filters[3])
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(filters[2])
        self.do41d = nn.Dropout2d(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(
            filters[2], filters[2], kernel_size=3, padding=1, stride=2, output_padding=1
        )

        self.conv33d = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(filters[2])
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(filters[2])
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(filters[1])
        self.do31d = nn.Dropout2d(p=0.2)

        self.upconv2 = nn.ConvTranspose2d(
            filters[1], filters[1], kernel_size=3, padding=1, stride=2, output_padding=1
        )

        self.conv22d = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(filters[1])
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(filters[0])
        self.do21d = nn.Dropout2d(p=0.2)

        self.upconv1 = nn.ConvTranspose2d(
            filters[0], filters[0], kernel_size=3, padding=1, stride=2, output_padding=1
        )

        self.conv12d = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(filters[0])
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(filters[0], filters[0], kernel_size=3, padding=1)

    def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor]:
        x = torch.cat((x1, x2), 1)
        # Encoder
        x11 = self.do11(F.relu(self.bn11(self.conv11(x))))
        x12 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12, kernel_size=2, stride=2)

        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22, kernel_size=2, stride=2)

        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33, kernel_size=2, stride=2)

        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43, kernel_size=2, stride=2)

        # Decoder with skip connections
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43.size(3) - x4d.size(3), 0, x43.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), x43), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33.size(3) - x3d.size(3), 0, x33.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), x33), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22.size(3) - x2d.size(3), 0, x22.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), x22), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12.size(3) - x1d.size(3), 0, x12.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), x12), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        return (x11d,)


@register_model("FC_Siam_diff")
class FC_Siam_diff(nn.Module):
    """FC-Siam-diff: siamese encoder with feature-difference skip connections.

    Args:
        in_channels: Channels of each input image (3 for RGB).
        base_channel: Width of the first stage.

    Forward returns a 1-tuple with a ``(B, base_channel, H, W)`` feature
    map at input resolution.
    """

    def __init__(self, in_channels: int, base_channel: int = 16) -> None:
        super().__init__()

        filters = [
            base_channel,
            base_channel * 2,
            base_channel * 4,
            base_channel * 8,
            base_channel * 16,
        ]

        self.conv11 = nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(filters[0])
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(filters[0])
        self.do12 = nn.Dropout2d(p=0.2)

        self.conv21 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(filters[1])
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(filters[1])
        self.do22 = nn.Dropout2d(p=0.2)

        self.conv31 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(filters[2])
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(filters[2])
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(filters[2])
        self.do33 = nn.Dropout2d(p=0.2)

        self.conv41 = nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(filters[3])
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(filters[3])
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(filters[3])
        self.do43 = nn.Dropout2d(p=0.2)

        self.upconv4 = nn.ConvTranspose2d(
            filters[3], filters[3], kernel_size=3, padding=1, stride=2, output_padding=1
        )

        self.conv43d = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(filters[3])
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(filters[3])
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(filters[2])
        self.do41d = nn.Dropout2d(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(
            filters[2], filters[2], kernel_size=3, padding=1, stride=2, output_padding=1
        )

        self.conv33d = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(filters[2])
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(filters[2])
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(filters[1])
        self.do31d = nn.Dropout2d(p=0.2)

        self.upconv2 = nn.ConvTranspose2d(
            filters[1], filters[1], kernel_size=3, padding=1, stride=2, output_padding=1
        )

        self.conv22d = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(filters[1])
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(filters[0])
        self.do21d = nn.Dropout2d(p=0.2)

        self.upconv1 = nn.ConvTranspose2d(
            filters[0], filters[0], kernel_size=3, padding=1, stride=2, output_padding=1
        )

        self.conv12d = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(filters[0])
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(filters[0], filters[0], kernel_size=3, padding=1)

    def _encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run one siamese branch, returning stage outputs and the bottleneck."""
        x11 = self.do11(F.relu(self.bn11(self.conv11(x))))
        x12 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12, kernel_size=2, stride=2)

        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22, kernel_size=2, stride=2)

        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33, kernel_size=2, stride=2)

        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43, kernel_size=2, stride=2)

        return x12, x22, x33, x43, x4p

    def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor]:
        x12_1, x22_1, x33_1, x43_1, _ = self._encode(x1)
        x12_2, x22_2, x33_2, x43_2, x4p = self._encode(x2)

        # Decoder: skip connections carry |branch1 - branch2|.
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), torch.abs(x43_1 - x43_2)), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), torch.abs(x33_1 - x33_2)), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), torch.abs(x22_1 - x22_2)), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), torch.abs(x12_1 - x12_2)), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        return (x11d,)


@register_model("FC_Siam_conc")
class FC_Siam_conc(nn.Module):
    """FC-Siam-conc: siamese encoder with feature-concatenation skip connections.

    Args:
        in_channels: Channels of each input image (3 for RGB).
        base_channel: Width of the first stage.

    Forward returns a 1-tuple with a ``(B, base_channel, H, W)`` feature
    map at input resolution.
    """

    def __init__(self, in_channels: int, base_channel: int = 16) -> None:
        super().__init__()

        filters = [
            base_channel,
            base_channel * 2,
            base_channel * 4,
            base_channel * 8,
            base_channel * 16,
        ]

        self.conv11 = nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(filters[0])
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(filters[0])
        self.do12 = nn.Dropout2d(p=0.2)

        self.conv21 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(filters[1])
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(filters[1])
        self.do22 = nn.Dropout2d(p=0.2)

        self.conv31 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(filters[2])
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(filters[2])
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(filters[2])
        self.do33 = nn.Dropout2d(p=0.2)

        self.conv41 = nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(filters[3])
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(filters[3])
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(filters[3])
        self.do43 = nn.Dropout2d(p=0.2)

        self.upconv4 = nn.ConvTranspose2d(
            filters[3], filters[3], kernel_size=3, padding=1, stride=2, output_padding=1
        )

        self.conv43d = nn.ConvTranspose2d(
            filters[3] + filters[4], filters[3], kernel_size=3, padding=1
        )
        self.bn43d = nn.BatchNorm2d(filters[3])
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(filters[3])
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(filters[2])
        self.do41d = nn.Dropout2d(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(
            filters[2], filters[2], kernel_size=3, padding=1, stride=2, output_padding=1
        )

        self.conv33d = nn.ConvTranspose2d(
            filters[2] + filters[3], filters[2], kernel_size=3, padding=1
        )
        self.bn33d = nn.BatchNorm2d(filters[2])
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(filters[2])
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(filters[1])
        self.do31d = nn.Dropout2d(p=0.2)

        self.upconv2 = nn.ConvTranspose2d(
            filters[1], filters[1], kernel_size=3, padding=1, stride=2, output_padding=1
        )

        self.conv22d = nn.ConvTranspose2d(
            filters[1] + filters[2], filters[1], kernel_size=3, padding=1
        )
        self.bn22d = nn.BatchNorm2d(filters[1])
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(filters[0])
        self.do21d = nn.Dropout2d(p=0.2)

        self.upconv1 = nn.ConvTranspose2d(
            filters[0], filters[0], kernel_size=3, padding=1, stride=2, output_padding=1
        )

        self.conv12d = nn.ConvTranspose2d(
            filters[0] + filters[1], filters[0], kernel_size=3, padding=1
        )
        self.bn12d = nn.BatchNorm2d(filters[0])
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(filters[0], filters[0], kernel_size=3, padding=1)

    def _encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run one siamese branch, returning stage outputs and the bottleneck."""
        x11 = self.do11(F.relu(self.bn11(self.conv11(x))))
        x12 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12, kernel_size=2, stride=2)

        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22, kernel_size=2, stride=2)

        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33, kernel_size=2, stride=2)

        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43, kernel_size=2, stride=2)

        return x12, x22, x33, x43, x4p

    def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor]:
        x12_1, x22_1, x33_1, x43_1, _ = self._encode(x1)
        x12_2, x22_2, x33_2, x43_2, x4p = self._encode(x2)

        # Decoder: skip connections carry both branches concatenated.
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), x43_1, x43_2), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), x33_1, x33_2), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), x22_1, x22_2), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), x12_1, x12_2), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        return (x11d,)
