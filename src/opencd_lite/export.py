"""ONNX export utilities.

Exports a :class:`~opencd_lite.inference.ChangeDetector` (or any module
taking two image tensors) so that applications can run inference with
``onnxruntime`` alone — no PyTorch at deployment time.

Note that the exported graph is the single ``forward`` pass; test-time
padding, sliding-window inference and binarization from
:class:`ChangeDetector` must be reproduced by the consuming application
(or the input size fixed to the training crop size, which is the common
deployment setup).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

__all__ = ["export_onnx", "verify_onnx"]

logger = logging.getLogger(__name__)


def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    *,
    input_size: tuple[int, int] = (256, 256),
    opset: int = 17,
    dynamo: bool = True,
    verify: bool = True,
    atol: float = 1e-4,
) -> Path:
    """Export a bi-temporal change detection model to ONNX.

    Args:
        model: Module whose ``forward(x1, x2)`` returns prediction logits.
        output_path: Destination ``.onnx`` file.
        input_size: Spatial size ``(H, W)`` of the exported graph. Must be
            divisible by the model's downsampling factor (32 covers all
            supported models).
        opset: ONNX opset version.
        dynamo: Use the dynamo-based exporter (PyTorch's recommended
            path). Falls back to the legacy TorchScript exporter if the
            dynamo export fails.
        verify: Run the exported model under onnxruntime and compare with
            the PyTorch output.
        atol: Absolute tolerance for verification.

    Returns:
        The path of the written ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    height, width = input_size
    example = (
        torch.randn(1, 3, height, width),
        torch.randn(1, 3, height, width),
    )
    try:
        _export(model, example, output_path, opset=opset, dynamo=dynamo)
    finally:
        model.train(was_training)

    # Embed the test-time protocol so ONNXChangeDetector.from_file can
    # reproduce padding / slide inference / binarization with no config.
    inference_cfg = getattr(model, "inference_cfg", None)
    if inference_cfg is not None:
        _embed_inference_metadata(output_path, inference_cfg)

    if verify:
        verify_onnx(model, output_path, example, atol=atol)
    return output_path


def _embed_inference_metadata(output_path: Path, inference_cfg: object) -> None:
    """Write the model's :class:`InferenceConfig` into the ONNX metadata."""
    import json

    import onnx

    from .onnx import ONNX_METADATA_KEY

    to_dict = getattr(inference_cfg, "to_dict", None)
    if to_dict is None:
        return
    proto = onnx.load(str(output_path))
    entry = proto.metadata_props.add()
    entry.key = ONNX_METADATA_KEY
    entry.value = json.dumps(to_dict())
    onnx.save(proto, str(output_path))


def _export(
    model: nn.Module,
    example: tuple[torch.Tensor, torch.Tensor],
    output_path: Path,
    *,
    opset: int,
    dynamo: bool,
) -> None:
    def run(use_dynamo: bool) -> None:
        torch.onnx.export(
            model,
            example,
            str(output_path),
            input_names=["image_from", "image_to"],
            output_names=["logits"],
            opset_version=opset,
            dynamo=use_dynamo,
        )

    # nn.MultiheadAttention's fused inference kernel
    # (aten::_native_multi_head_attention) has no ONNX lowering; force
    # the decomposed path while tracing (transformer-based models).
    mha_backend = getattr(torch.backends, "mha", None)
    fastpath_was_enabled = mha_backend.get_fastpath_enabled() if mha_backend is not None else None
    if mha_backend is not None:
        mha_backend.set_fastpath_enabled(False)
    try:
        if dynamo:
            try:
                run(use_dynamo=True)
                return
            except Exception:  # noqa: BLE001 - deliberate fallback path
                logger.warning(
                    "Dynamo ONNX export failed; falling back to the legacy exporter",
                    exc_info=True,
                )
        run(use_dynamo=False)
    finally:
        if mha_backend is not None and fastpath_was_enabled is not None:
            mha_backend.set_fastpath_enabled(fastpath_was_enabled)


def verify_onnx(
    model: nn.Module,
    onnx_path: str | Path,
    example: tuple[torch.Tensor, torch.Tensor],
    *,
    atol: float = 1e-4,
) -> float:
    """Compare onnxruntime output against PyTorch output.

    Returns:
        The maximum absolute difference.

    Raises:
        AssertionError: If outputs differ by more than ``atol``.
    """
    import onnxruntime as ort  # imported lazily: only needed with the "export" extra

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            expected = model(*example)
    finally:
        model.train(was_training)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in session.get_inputs()]
    feeds = {name: tensor.numpy() for name, tensor in zip(input_names, example, strict=True)}
    (actual,) = session.run(None, feeds)

    max_diff = float((expected - torch.from_numpy(actual)).abs().max())
    assert max_diff <= atol, (
        f"ONNX output differs from PyTorch output: max abs diff {max_diff:.3e} > atol {atol:.0e}"
    )
    return max_diff
