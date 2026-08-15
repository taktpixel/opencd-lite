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


def test_export_bit_to_onnx(configs_dir: Path, tmp_path: Path) -> None:
    """BIT: siamese backbone + neck + transformer head export end to end."""
    from opencd_lite import build_model

    detector = build_model(configs_dir / "bit" / "bit_r18_256x256_40k_levircd.py")
    onnx_path = export_onnx(
        detector,
        tmp_path / "bit.onnx",
        input_size=(64, 64),
        verify=True,
        atol=1e-4,
    )
    assert onnx_path.is_file()


def test_export_changer_to_onnx(configs_dir: Path, tmp_path: Path) -> None:
    """Changer: interaction backbone + multi-input head (incl. grid_sample)."""
    from opencd_lite import build_model

    detector = build_model(configs_dir / "changer" / "changer_ex_r18_512x512_40k_levircd.py")
    onnx_path = export_onnx(
        detector,
        tmp_path / "changer.onnx",
        input_size=(64, 64),
        verify=True,
        atol=1e-4,
    )
    assert onnx_path.is_file()


def test_export_stanet_to_onnx(configs_dir: Path, tmp_path: Path) -> None:
    """STANet: pyramid attention head with metric (distance) output."""
    from opencd_lite import build_model

    detector = build_model(configs_dir / "stanet" / "stanet_pam_256x256_40k_levircd.py")
    onnx_path = export_onnx(
        detector,
        tmp_path / "stanet.onnx",
        input_size=(64, 64),
        verify=True,
        atol=1e-4,
    )
    assert onnx_path.is_file()


def test_export_lightcdnet_to_onnx(configs_dir: Path, tmp_path: Path) -> None:
    """LightCDNet: criss-cross attention + TinyFPN neck export end to end."""
    from opencd_lite import build_model

    detector = build_model(configs_dir / "lightcdnet" / "lightcdnet_s_256x256_40k_levircd.py")
    onnx_path = export_onnx(
        detector,
        tmp_path / "lightcdnet.onnx",
        input_size=(64, 64),
        verify=True,
        atol=1e-4,
    )
    assert onnx_path.is_file()


def test_export_changeformer_to_onnx(configs_dir: Path, tmp_path: Path) -> None:
    """ChangeFormer: siamese MiT transformer backbone + SegFormer head."""
    from opencd_lite import build_model

    detector = build_model(
        configs_dir / "changeformer" / "changeformer_mit-b0_256x256_40k_levircd.py"
    )
    onnx_path = export_onnx(
        detector,
        tmp_path / "changeformer.onnx",
        input_size=(64, 64),
        verify=True,
        atol=1e-4,
    )
    assert onnx_path.is_file()


def test_export_changestar_to_onnx(configs_dir: Path, tmp_path: Path) -> None:
    """ChangeStar: FarSeg FPN neck + scene embedding + ChangeMixin head."""
    from opencd_lite import build_model

    detector = build_model(
        configs_dir / "changestar" / "changestar_farseg_1x96_256x256_40k_levircd.py"
    )
    onnx_path = export_onnx(
        detector,
        tmp_path / "changestar.onnx",
        input_size=(64, 64),
        verify=True,
        atol=1e-4,
    )
    assert onnx_path.is_file()


def test_export_tinycd_v2_to_onnx(configs_dir: Path, tmp_path: Path) -> None:
    """TinyCD v2: TinyNet backbone + TinyBlock FPN + priori-attn head."""
    from opencd_lite import build_model

    detector = build_model(configs_dir / "tinycd_v2" / "tinycd_v2_s_256x256_40k_levircd.py")
    onnx_path = export_onnx(
        detector,
        tmp_path / "tinycd_v2.onnx",
        input_size=(64, 64),
        verify=True,
        atol=1e-4,
    )
    assert onnx_path.is_file()


def test_export_ban_to_onnx(tmp_path: Path, make_small_ban_head) -> None:
    """BAN: frozen ViT tower + adapter head export end to end."""
    from opencd_lite import BANChangeDetector
    from opencd_lite.models import VisionTransformer

    detector = BANChangeDetector(
        image_encoder=VisionTransformer(
            img_size=(16, 16),
            patch_size=4,
            embed_dims=24,
            num_layers=2,
            num_heads=2,
            out_indices=(1,),
            pre_norm=True,
            output_cls_token=True,
            frozen_exclude=[],
        ),
        decode_head=make_small_ban_head(),
        encoder_resolution={"size": (16, 16), "mode": "bilinear"},
    )
    onnx_path = export_onnx(
        detector,
        tmp_path / "ban.onnx",
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
