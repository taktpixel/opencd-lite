"""Test-time inference protocol (torch-free).

:class:`InferenceConfig` describes how a model's raw output is turned
into a change mask — the mmseg test-time behavior — independently of any
framework. Both the PyTorch :class:`~opencd_lite.inference.ChangeDetector`
and the torch-free :class:`~opencd_lite.onnx.ONNXChangeDetector` consume
it, so it lives in its own module with no torch dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["InferenceConfig"]


@dataclass(frozen=True)
class InferenceConfig:
    """Test-time protocol derived from an Open-CD config.

    Attributes:
        mode: ``"whole"`` (single forward) or ``"slide"``
            (sliding-window with logit averaging), as in mmseg.
        crop_size: Window size ``(h, w)`` for slide mode.
        stride: Window stride ``(h, w)`` for slide mode.
        out_index: Which element of the model's output tuple is the
            prediction (Open-CD ``decode_head.in_index``).
        out_channels: Number of prediction channels; 1 means binary
            sigmoid output.
        threshold: Binarization threshold applied to the sigmoid output
            when ``out_channels == 1`` (mmseg defaults to 0.3 when a
            config leaves it unset).
    """

    mode: str = "whole"
    crop_size: tuple[int, int] | None = None
    stride: tuple[int, int] | None = None
    out_index: int = -1
    out_channels: int = 1
    threshold: float = 0.3

    def __post_init__(self) -> None:
        if self.mode not in ("whole", "slide"):
            raise ValueError(f"Unsupported inference mode: {self.mode!r}")
        if self.mode == "slide" and (self.crop_size is None or self.stride is None):
            raise ValueError("Slide mode requires both crop_size and stride")

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form, e.g. for embedding in ONNX metadata."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferenceConfig:
        """Rebuild from :meth:`to_dict`, restoring the tuple fields."""
        crop_size = data.get("crop_size")
        stride = data.get("stride")
        return cls(
            mode=data.get("mode", "whole"),
            crop_size=tuple(crop_size) if crop_size is not None else None,
            stride=tuple(stride) if stride is not None else None,
            out_index=data.get("out_index", -1),
            out_channels=data.get("out_channels", 1),
            threshold=data.get("threshold", 0.3),
        )
