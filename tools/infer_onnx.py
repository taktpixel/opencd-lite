#!/usr/bin/env python
"""Run change detection inference from an exported ONNX graph — torch-free.

Reads a "before"/"after" image pair, runs the model exported by
``tools/export.py`` with ``onnxruntime``, and writes the binary change
mask. No PyTorch is required — only ``pip install opencd-lite[onnx]``.

The inference protocol (whole vs. sliding-window, crop size, stride,
threshold) is read from the ONNX metadata written at export time; the
overrides below are for ONNX files produced elsewhere.

Example:
    python tools/infer_onnx.py cgnet.onnx before.png after.png -o mask.png

Requires the ``onnx`` extra: ``pip install opencd-lite[onnx]``.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from opencd_lite import InferenceConfig
from opencd_lite.onnx import ONNXChangeDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path, help="exported .onnx graph")
    parser.add_argument("before", type=Path, help="'before' image")
    parser.add_argument("after", type=Path, help="'after' image")
    parser.add_argument(
        "--output", "-o", type=Path, required=True, help="destination mask image (PNG)"
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="scale the mask 0/1 -> 0/255 so it is visible as an image",
    )
    # Optional overrides for ONNX files without embedded metadata.
    parser.add_argument("--mode", choices=("whole", "slide"), default=None)
    parser.add_argument("--crop-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--stride", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def _apply_overrides(base: InferenceConfig, args: argparse.Namespace) -> InferenceConfig:
    """Merge only the CLI flags the user set onto the (embedded) config."""
    changes: dict[str, object] = {}
    if args.mode is not None:
        changes["mode"] = args.mode
    if args.crop_size is not None:
        changes["crop_size"] = tuple(args.crop_size)
    if args.stride is not None:
        changes["stride"] = tuple(args.stride)
    if args.threshold is not None:
        changes["threshold"] = args.threshold
    return dataclasses.replace(base, **changes) if changes else base


def _read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def main() -> None:
    args = parse_args()
    has_overrides = any((args.mode, args.crop_size, args.stride, args.threshold is not None))
    if has_overrides:
        # Start from the embedded protocol when present, then merge only
        # the flags the user set, so a partial override (e.g. just
        # --threshold) keeps the graph's slide crop/stride.
        try:
            base = ONNXChangeDetector.from_file(args.onnx).inference_cfg
        except ValueError:
            base = InferenceConfig()
        detector = ONNXChangeDetector(args.onnx, inference=_apply_overrides(base, args))
    else:
        detector = ONNXChangeDetector.from_file(args.onnx)
    before = _read_rgb(args.before)
    after = _read_rgb(args.after)
    mask = detector.predict(before, after)

    from PIL import Image

    out = (mask * 255).astype(np.uint8) if args.scale else mask
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="L").save(args.output)
    changed = int(mask.sum())
    print(
        f"{args.onnx.name}: {detector.inference_cfg.mode} inference on {before.shape[1]}x"
        f"{before.shape[0]} -> {args.output} ({changed} changed pixels, "
        f"{changed / mask.size * 100:.3f}%)"
    )


if __name__ == "__main__":
    main()
