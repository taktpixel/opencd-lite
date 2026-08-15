"""Parameter-free necks fusing bi-temporal backbone features.

Adapted from Open-CD (https://github.com/likyoo/open-cd, Apache-2.0).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

__all__ = ["FeatureFusionNeck"]


class FeatureFusionNeck(nn.Module):
    """Fuse per-stage feature pairs from a siamese backbone.

    This neck holds no parameters; it only combines the two temporal
    feature pyramids, so it contributes no checkpoint keys (matching the
    upstream Open-CD checkpoints, which store none for it).

    Args:
        policy: Fusion operation: ``'concat'``, ``'sum'``, ``'diff'`` or
            ``'abs_diff'``.
        in_channels: Unused; accepted for config compatibility.
        channels: Unused; accepted for config compatibility.
        out_indices: Indices of the fused stages to return.
    """

    _POLICIES = ("concat", "sum", "diff", "abs_diff")

    def __init__(
        self,
        policy: str = "concat",
        in_channels: Sequence[int] | None = None,
        channels: int | None = None,
        out_indices: Sequence[int] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        if policy not in self._POLICIES:
            raise ValueError(f"Unknown fusion policy {policy!r} (supported: {self._POLICIES})")
        self.policy = policy
        self.out_indices = tuple(out_indices)

    @staticmethod
    def fusion(x1: Tensor, x2: Tensor, policy: str) -> Tensor:
        """Fuse one feature pair according to ``policy``."""
        if policy == "concat":
            return torch.cat([x1, x2], dim=1)
        if policy == "sum":
            return x1 + x2
        if policy == "diff":
            return x2 - x1
        if policy == "abs_diff":
            return torch.abs(x1 - x2)
        raise ValueError(f"Unknown fusion policy {policy!r}")

    def forward(self, x1: Sequence[Tensor], x2: Sequence[Tensor]) -> tuple[Tensor, ...]:
        if len(x1) != len(x2):
            raise ValueError("Feature pyramids must have the same number of stages")
        outs = [self.fusion(f1, f2, self.policy) for f1, f2 in zip(x1, x2, strict=True)]
        return tuple(outs[i] for i in self.out_indices)
