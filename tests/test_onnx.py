"""Torch-free ONNX inference tests (require the "onnx" and "export" extras).

These pin the contract that matters for deployment: the numpy/onnxruntime
:class:`~opencd_lite.onnx.ONNXChangeDetector` reproduces the PyTorch
:class:`~opencd_lite.inference.ChangeDetector` — matching its masks on
the tested inputs — and the ONNX inference path imports without PyTorch
installed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from opencd_lite import IMAGENET_SPEC, InferenceConfig
from opencd_lite.protocol import InferenceConfig as ProtocolInferenceConfig
from opencd_lite.transforms import normalize_image_numpy


def _random_image(height: int, width: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


# --------------------------------------------------------------------------
# Torch-free pieces: no onnxruntime / torch needed
# --------------------------------------------------------------------------
def test_inference_config_roundtrips_through_dict() -> None:
    cfg = InferenceConfig(mode="slide", crop_size=(256, 256), stride=(128, 128), threshold=0.5)
    restored = ProtocolInferenceConfig.from_dict(cfg.to_dict())
    assert restored == cfg
    assert isinstance(restored.crop_size, tuple)


def test_normalize_image_numpy_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    from opencd_lite.transforms import normalize_image

    image = _random_image(40, 30)
    expected = normalize_image(image, IMAGENET_SPEC)
    actual = normalize_image_numpy(image, IMAGENET_SPEC)
    assert actual.shape == (3, 40, 30)
    # Op-for-op identical (same float32 mean/std, same order): bit-exact.
    np.testing.assert_array_equal(actual, expected.numpy())
    _ = torch  # importorskip guard only


def test_onnx_inference_path_imports_without_torch() -> None:
    """`import opencd_lite` and the ONNX API must not require torch."""
    script = textwrap.dedent(
        """
        import sys
        # Simulate a torch-free environment: any torch import fails.
        for name in list(sys.modules):
            if name == "torch" or name.startswith("torch."):
                del sys.modules[name]
        import builtins
        real_import = builtins.__import__
        def guard(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch is not available")
            return real_import(name, *args, **kwargs)
        builtins.__import__ = guard

        import opencd_lite
        from opencd_lite import ONNXChangeDetector, InferenceConfig, normalize_image_numpy
        from opencd_lite.onnx import ONNXChangeDetector as _Direct
        from opencd_lite.transforms import IMAGENET_SPEC, normalize_image_numpy as _n
        import numpy as np
        # A representative preprocessing call must run with no torch present.
        out = normalize_image_numpy(np.zeros((8, 8, 3), dtype=np.uint8))
        assert out.shape == (3, 8, 8)
        assert "torch" not in sys.modules
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("OK")


# --------------------------------------------------------------------------
# Full round-trip: export (torch) then infer (numpy/onnxruntime)
#
# These require onnxruntime, so they carry the ``onnx`` marker
# individually — the torch-free tests above must still run (and gate
# regressions) in an environment without the extra, and under
# ``pytest -m "not onnx"``.
# --------------------------------------------------------------------------
onnx = pytest.mark.onnx


@pytest.fixture()
def exported_slide_cgnet(cgnet_small, tmp_path: Path):
    """Export a small CGNet as a slide-mode ONNX graph and return its path + detector."""
    pytest.importorskip("onnxruntime")
    from opencd_lite import ChangeDetector, export_onnx

    detector = ChangeDetector(
        cgnet_small,
        inference=InferenceConfig(mode="slide", crop_size=(64, 64), stride=(32, 32), threshold=0.5),
    )
    path = export_onnx(detector, tmp_path / "cgnet.onnx", input_size=(64, 64), verify=False)
    return path, detector


@onnx
def test_export_embeds_inference_metadata(exported_slide_cgnet) -> None:
    from opencd_lite.onnx import ONNXChangeDetector

    path, _ = exported_slide_cgnet
    onnx_detector = ONNXChangeDetector.from_file(path)
    assert onnx_detector.inference_cfg.mode == "slide"
    assert onnx_detector.inference_cfg.crop_size == (64, 64)
    assert onnx_detector.inference_cfg.stride == (32, 32)
    assert onnx_detector.inference_cfg.threshold == 0.5


@onnx
@pytest.mark.parametrize("size", [(64, 64), (70, 60), (100, 90), (96, 96)])
def test_onnx_slide_matches_pytorch(exported_slide_cgnet, size) -> None:
    """The torch-free slide inference must equal the PyTorch mask exactly."""
    from opencd_lite.onnx import ONNXChangeDetector

    path, detector = exported_slide_cgnet
    onnx_detector = ONNXChangeDetector.from_file(path)
    before = _random_image(*size, seed=1)
    after = _random_image(*size, seed=2)

    onnx_mask = onnx_detector.predict(before, after)
    torch_mask = detector.predict(before, after)
    assert onnx_mask.shape == size
    assert onnx_mask.dtype == np.uint8
    np.testing.assert_array_equal(onnx_mask, torch_mask)


@onnx
def test_onnx_whole_matches_pytorch_at_export_size(cgnet_small, tmp_path: Path) -> None:
    """Whole-mode graphs are fixed-size; parity holds at the exported size."""
    from opencd_lite import ChangeDetector, export_onnx
    from opencd_lite.onnx import ONNXChangeDetector

    detector = ChangeDetector(cgnet_small, inference=InferenceConfig(mode="whole", threshold=0.5))
    path = export_onnx(detector, tmp_path / "cgnet_whole.onnx", input_size=(64, 64), verify=False)
    onnx_detector = ONNXChangeDetector.from_file(path)

    before, after = _random_image(64, 64, seed=1), _random_image(64, 64, seed=2)
    np.testing.assert_array_equal(
        onnx_detector.predict(before, after), detector.predict(before, after)
    )


@onnx
def test_explicit_inference_config_overrides_metadata(exported_slide_cgnet) -> None:
    from opencd_lite.onnx import ONNXChangeDetector

    path, _ = exported_slide_cgnet
    override = InferenceConfig(mode="whole", threshold=0.9)
    onnx_detector = ONNXChangeDetector(path, inference=override)
    assert onnx_detector.inference_cfg.mode == "whole"
    assert onnx_detector.inference_cfg.threshold == 0.9


@onnx
def test_wrong_size_whole_input_raises_clear_error(cgnet_small, tmp_path: Path) -> None:
    """A fixed-size graph fed the wrong size fails with a helpful message, not ORT noise."""
    from opencd_lite import ChangeDetector, export_onnx
    from opencd_lite.onnx import ONNXChangeDetector

    detector = ChangeDetector(cgnet_small, inference=InferenceConfig(mode="whole"))
    path = export_onnx(detector, tmp_path / "cgnet_whole.onnx", input_size=(64, 64), verify=False)
    onnx_detector = ONNXChangeDetector.from_file(path)
    with pytest.raises(ValueError, match="fixed-size"):
        onnx_detector.predict(_random_image(96, 96), _random_image(96, 96))


@onnx
def test_missing_metadata_requires_explicit_config(cgnet_small, tmp_path: Path) -> None:
    """from_file on a graph without embedded protocol raises a clear error."""
    import onnx as onnx_mod

    from opencd_lite import ChangeDetector, export_onnx
    from opencd_lite.onnx import ONNX_METADATA_KEY, ONNXChangeDetector

    detector = ChangeDetector(cgnet_small, inference=InferenceConfig(mode="whole"))
    path = export_onnx(detector, tmp_path / "cgnet.onnx", input_size=(64, 64), verify=False)
    # Strip the embedded metadata.
    proto = onnx_mod.load(str(path))
    kept = [p for p in proto.metadata_props if p.key != ONNX_METADATA_KEY]
    del proto.metadata_props[:]
    proto.metadata_props.extend(kept)
    onnx_mod.save(proto, str(path))

    with pytest.raises(ValueError, match="no embedded inference metadata"):
        ONNXChangeDetector.from_file(path)
    # But an explicit config still works.
    detector_ok = ONNXChangeDetector(path, inference=InferenceConfig(mode="whole"))
    assert detector_ok.inference_cfg.mode == "whole"


@onnx
def test_two_class_head_uses_argmax(tmp_path: Path) -> None:
    """A 2-channel (softmax) head is binarized by argmax, not a threshold."""
    from opencd_lite import ChangeDetector, SNUNet_ECAM, export_onnx
    from opencd_lite.models import ConvSegHead
    from opencd_lite.onnx import ONNXChangeDetector

    detector = ChangeDetector(
        SNUNet_ECAM(in_channels=3, base_channel=16),
        inference=InferenceConfig(mode="whole", out_channels=2),
        decode_head=ConvSegHead(in_channels=64, num_classes=2),
    )
    path = export_onnx(detector, tmp_path / "snunet.onnx", input_size=(64, 64), verify=False)
    onnx_detector = ONNXChangeDetector.from_file(path)
    assert onnx_detector.inference_cfg.out_channels == 2

    before, after = _random_image(64, 64, seed=3), _random_image(64, 64, seed=4)
    mask = onnx_detector.predict(before, after)
    assert set(np.unique(mask)) <= {0, 1}
    np.testing.assert_array_equal(mask, detector.predict(before, after))
