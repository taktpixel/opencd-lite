#!/usr/bin/env python
"""Export an Open-CD model to ONNX for torch-free deployment.

Builds a :class:`~opencd_lite.inference.ChangeDetector` from an Open-CD
config (optionally loading published weights) and writes an ONNX graph
with the test-time protocol embedded in its metadata, so
``opencd_lite.onnx.ONNXChangeDetector`` — and ``tools/infer_onnx.py`` —
can run it with only ``onnxruntime`` and ``numpy``.

Example:
    python tools/export.py configs/cgnet/cgnet_256x256_40k_levircd.py \
        --checkpoint cgnet_levircd.pth --output cgnet.onnx

Requires the ``export`` extra: ``pip install opencd-lite[export]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from opencd_lite import build_model, export_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Open-CD compatible config file")
    parser.add_argument("--checkpoint", type=Path, default=None, help="weights to load")
    parser.add_argument("--output", "-o", type=Path, required=True, help="destination .onnx file")
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=(256, 256),
        metavar=("H", "W"),
        help="spatial size of the exported graph (must be a multiple of 32)",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the onnxruntime-vs-PyTorch verification pass",
    )
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {"pretrained": False} if args.checkpoint is None else None
    detector = build_model(args.config, checkpoint=args.checkpoint, backbone_overrides=overrides)
    path = export_onnx(
        detector,
        args.output,
        input_size=tuple(args.input_size),
        opset=args.opset,
        verify=not args.no_verify,
        atol=args.atol,
    )
    cfg = detector.inference_cfg
    print(
        f"Exported {cfg.mode} model to {path} "
        f"(input {tuple(args.input_size)}, threshold {cfg.threshold}, "
        f"out_channels {cfg.out_channels}); inference protocol embedded in metadata."
    )


if __name__ == "__main__":
    main()
