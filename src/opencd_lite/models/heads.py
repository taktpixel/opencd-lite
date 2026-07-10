"""Parametric decode heads.

Most supported models are fully self-contained and use Open-CD's
parameter-free ``IdentityHead``; those need no module here. Models
configured with ``mmseg.FCNHead(num_convs=0, concat_input=False)``
(FC-Siam family, SNUNet) classify through a learned 1x1 convolution
whose weights live in the checkpoint under ``decode_head.conv_seg.*``.
:class:`ConvSegHead` reproduces exactly that reduced FCNHead.
"""

from __future__ import annotations

from torch import Tensor, nn

__all__ = ["ConvSegHead"]


class ConvSegHead(nn.Module):
    """Dropout + 1x1 convolution classifier (mmseg ``FCNHead(num_convs=0)``).

    Args:
        in_channels: Channels of the incoming feature map.
        num_classes: Number of output classes (logit channels).
        dropout_ratio: Dropout applied before the classifier during
            training (mmseg default 0.1).

    The attribute is named ``conv_seg`` to match Open-CD checkpoint keys.
    """

    def __init__(self, in_channels: int, num_classes: int, dropout_ratio: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        return self.conv_seg(x)
