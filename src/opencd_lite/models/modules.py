"""Small building blocks shared by ported model implementations.

:class:`ConvModule` is a minimal stand-in for ``mmcv.cnn.ConvModule``
in its default ``conv -> norm -> act`` order. Submodule attribute names
(``conv``, ``bn``) match mmcv so that checkpoint keys map one-to-one.
"""

from __future__ import annotations

from torch import Tensor, nn

__all__ = ["ConvModule"]


class ConvModule(nn.Module):
    """Convolution with optional BatchNorm and ReLU (mmcv naming).

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        dilation: Convolution dilation.
        groups: Convolution groups.
        norm: Insert a BatchNorm2d between convolution and activation.
            Mirrors mmcv's ``norm_cfg=dict(type='BN')``; the convolution
            bias is disabled automatically, as in mmcv's ``bias='auto'``.
        act: Apply a ReLU after the (normalized) convolution. Mirrors
            mmcv's default ``act_cfg=dict(type='ReLU')``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        norm: bool = False,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=not norm,
        )
        self.bn = nn.BatchNorm2d(out_channels) if norm else None
        self.activate = nn.ReLU(inplace=True) if act else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activate is not None:
            x = self.activate(x)
        return x
