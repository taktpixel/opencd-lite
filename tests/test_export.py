"""ONNX export smoke tests (require the "export" extra)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")
pytestmark = pytest.mark.export

from opencd_lite import ChangeDetector, export_onnx  # noqa: E402


def test_export_cgnet_to_onnx(cgnet_small, tmp_path: Path) -> None:
    detector = ChangeDetector(cgnet_small)
    onnx_path = export_onnx(
        detector,
        tmp_path / "cgnet.onnx",
        input_size=(64, 64),
        verify=True,
        atol=1e-4,
    )
    assert onnx_path.is_file()


def test_export_ifn_to_onnx(ifn_small, tmp_path: Path) -> None:
    detector = ChangeDetector(ifn_small)
    onnx_path = export_onnx(
        detector,
        tmp_path / "ifn.onnx",
        input_size=(64, 64),
        verify=True,
        atol=1e-4,
    )
    assert onnx_path.is_file()


def test_export_detector_with_decode_head(tmp_path: Path) -> None:
    """Models with a parametric head (here SNUNet) export including the head."""
    import onnxruntime as ort
    import torch

    from opencd_lite import SNUNet_ECAM
    from opencd_lite.models import ConvSegHead

    detector = ChangeDetector(
        SNUNet_ECAM(in_channels=3, base_channel=16),
        decode_head=ConvSegHead(in_channels=64, num_classes=2),
    )
    onnx_path = export_onnx(
        detector,
        tmp_path / "snunet.onnx",
        input_size=(64, 64),
        verify=True,
        atol=1e-4,
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (out,) = session.run(
        None,
        {
            "image_from": torch.randn(1, 3, 64, 64).numpy(),
            "image_to": torch.randn(1, 3, 64, 64).numpy(),
        },
    )
    # The exported graph must include the classification head (2 classes).
    assert out.shape == (1, 2, 64, 64)
