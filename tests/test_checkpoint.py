"""Round-trip tests for Open-CD checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from opencd_lite import CGNet, ChangeDetector, load_opencd_checkpoint
from opencd_lite.checkpoint import extract_backbone_state_dict


def _make_opencd_style_checkpoint(model: torch.nn.Module, path: Path) -> dict[str, int]:
    """Save a checkpoint in the Open-CD (mmengine) on-disk layout."""
    state_dict = {f"backbone.{k}": v.clone() for k, v in model.state_dict().items()}
    # Harness-only entries present in real Open-CD checkpoints.
    state_dict["decode_head.dummy"] = torch.zeros(1)
    state_dict["auxiliary_head.dummy"] = torch.zeros(1)
    checkpoint = {
        "meta": {"seed": 0, "dataset_meta": {"classes": ("unchanged", "changed")}},
        "state_dict": state_dict,
    }
    torch.save(checkpoint, path)
    return {"model_keys": len(state_dict) - 2}


def test_extract_backbone_state_dict(cgnet_small, tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pth"
    _make_opencd_style_checkpoint(cgnet_small, path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    state_dict, ignored = extract_backbone_state_dict(checkpoint)
    assert set(state_dict) == set(cgnet_small.state_dict())
    assert sorted(ignored) == ["auxiliary_head.dummy", "decode_head.dummy"]


def test_load_into_bare_model(cgnet_small, tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pth"
    _make_opencd_style_checkpoint(cgnet_small, path)

    fresh = CGNet(pretrained=False)
    report = load_opencd_checkpoint(fresh, path)
    assert not report.missing_keys
    assert not report.unexpected_keys

    for key, value in fresh.state_dict().items():
        assert torch.equal(value, cgnet_small.state_dict()[key]), key


def test_load_into_change_detector(cgnet_small, tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pth"
    _make_opencd_style_checkpoint(cgnet_small, path)

    detector = ChangeDetector(CGNet(pretrained=False))
    report = load_opencd_checkpoint(detector, path)
    assert not report.missing_keys
    assert torch.equal(
        detector.backbone.state_dict()["conv_reduce_1.conv.weight"],
        cgnet_small.state_dict()["conv_reduce_1.conv.weight"],
    )


def test_strict_load_rejects_mismatched_checkpoint(cgnet_small, tmp_path: Path) -> None:
    path = tmp_path / "bad.pth"
    torch.save({"state_dict": {"backbone.not_a_real_key": torch.zeros(1)}}, path)

    with pytest.raises(RuntimeError, match="does not match"):
        load_opencd_checkpoint(CGNet(pretrained=False), path)


def test_checkpoint_with_unimportable_harness_objects(cgnet_small, tmp_path: Path) -> None:
    """Real Open-CD checkpoints pickle mmengine objects; loading must survive them."""
    import sys
    import types

    # Fabricate a class from a module that will not exist at load time,
    # standing in for mmengine.logging.history_buffer.HistoryBuffer.
    fake_module = types.ModuleType("fake_mmengine_module")
    fake_class = type("HistoryBuffer", (), {"__module__": "fake_mmengine_module"})
    fake_module.HistoryBuffer = fake_class  # type: ignore[attr-defined]
    sys.modules["fake_mmengine_module"] = fake_module
    try:
        path = tmp_path / "with_harness_objects.pth"
        torch.save(
            {
                "meta": {"seed": 0},
                "message_hub": {"log_scalars": {"train/loss": fake_class()}},
                "state_dict": {
                    f"backbone.{k}": v.clone() for k, v in cgnet_small.state_dict().items()
                },
            },
            path,
        )
    finally:
        del sys.modules["fake_mmengine_module"]

    fresh = CGNet(pretrained=False)
    with pytest.warns(UserWarning, match="weights_only"):
        report = load_opencd_checkpoint(fresh, path)
    assert not report.missing_keys
    assert not report.unexpected_keys


def test_load_checkpoint_with_decode_head(configs_dir: Path, tmp_path: Path) -> None:
    """FC-Siam/SNUNet checkpoints carry decode_head.conv_seg.* weights."""
    from opencd_lite import build_model

    config = configs_dir / "fcsn" / "fc_ef_256x256_40k_levircd.py"
    source = build_model(config)
    state_dict = {
        **{f"backbone.{k}": v.clone() for k, v in source.backbone.state_dict().items()},
        **{f"decode_head.{k}": v.clone() for k, v in source.decode_head.state_dict().items()},
        "auxiliary_head.dummy": torch.zeros(1),
    }
    path = tmp_path / "fc_ef.pth"
    torch.save({"state_dict": state_dict}, path)

    fresh = build_model(config, checkpoint=path)
    assert torch.equal(fresh.decode_head.conv_seg.weight, source.decode_head.conv_seg.weight)
    assert torch.equal(
        fresh.backbone.state_dict()["conv11.weight"],
        source.backbone.state_dict()["conv11.weight"],
    )


def test_load_bit_checkpoint_round_trip(configs_dir: Path, tmp_path: Path) -> None:
    """BIT: siamese ResNet backbone + parametric BITHead, Open-CD key layout."""
    from opencd_lite import build_model

    config = configs_dir / "bit" / "bit_r18_256x256_40k_levircd.py"
    source = build_model(config)
    state_dict = {
        **{f"backbone.{k}": v.clone() for k, v in source.backbone.state_dict().items()},
        **{f"decode_head.{k}": v.clone() for k, v in source.decode_head.state_dict().items()},
    }
    path = tmp_path / "bit.pth"
    torch.save({"state_dict": state_dict}, path)

    fresh = build_model(config, checkpoint=path)
    assert torch.equal(fresh.decode_head.enc_pos_embedding, source.decode_head.enc_pos_embedding)
    assert torch.equal(
        fresh.backbone.state_dict()["stem.0.weight"],
        source.backbone.state_dict()["stem.0.weight"],
    )
    x1 = torch.randn(1, 3, 64, 64)
    x2 = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        assert torch.equal(fresh(x1, x2), source(x1, x2))


def test_load_changer_checkpoint_round_trip(configs_dir: Path, tmp_path: Path) -> None:
    """Changer: interaction backbone + multi-input FDAF head, Open-CD key layout."""
    from opencd_lite import build_model

    config = configs_dir / "changer" / "changer_ex_r18_512x512_40k_levircd.py"
    source = build_model(config)
    state_dict = {
        **{f"backbone.{k}": v.clone() for k, v in source.backbone.state_dict().items()},
        **{f"decode_head.{k}": v.clone() for k, v in source.decode_head.state_dict().items()},
    }
    path = tmp_path / "changer.pth"
    torch.save({"state_dict": state_dict}, path)

    fresh = build_model(config, checkpoint=path)
    assert torch.equal(
        fresh.decode_head.neck_layer.flow_make[0].weight,
        source.decode_head.neck_layer.flow_make[0].weight,
    )
    x1 = torch.randn(1, 3, 64, 64)
    x2 = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        assert torch.equal(fresh(x1, x2), source(x1, x2))


def test_load_stanet_checkpoint_round_trip(configs_dir: Path, tmp_path: Path) -> None:
    """STANet: siamese backbone + multi-input metric head, Open-CD key layout."""
    from opencd_lite import build_model

    config = configs_dir / "stanet" / "stanet_pam_256x256_40k_levircd.py"
    source = build_model(config)
    state_dict = {
        **{f"backbone.{k}": v.clone() for k, v in source.backbone.state_dict().items()},
        **{f"decode_head.{k}": v.clone() for k, v in source.decode_head.state_dict().items()},
    }
    path = tmp_path / "stanet.pth"
    torch.save({"state_dict": state_dict}, path)

    fresh = build_model(config, checkpoint=path)
    assert torch.equal(
        fresh.decode_head.netA.Self_Att.conv_bn[0].weight,
        source.decode_head.netA.Self_Att.conv_bn[0].weight,
    )
    x1 = torch.randn(1, 3, 64, 64)
    x2 = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        assert torch.equal(fresh(x1, x2), source(x1, x2))


def test_load_lightcdnet_checkpoint_round_trip(configs_dir: Path, tmp_path: Path) -> None:
    """LightCDNet: dual-input backbone + parametric TinyFPN neck + DS_FPNHead."""
    from opencd_lite import build_model

    config = configs_dir / "lightcdnet" / "lightcdnet_s_256x256_40k_levircd.py"
    source = build_model(config)
    state_dict = {
        **{f"backbone.{k}": v.clone() for k, v in source.backbone.state_dict().items()},
        **{f"neck.{k}": v.clone() for k, v in source.neck.state_dict().items()},
        **{f"decode_head.{k}": v.clone() for k, v in source.decode_head.state_dict().items()},
        "auxiliary_head.dummy": torch.zeros(1),
    }
    path = tmp_path / "lightcdnet.pth"
    torch.save({"state_dict": state_dict}, path)

    fresh = build_model(config, checkpoint=path)
    # neck.* keys are loaded (not ignored) for parametric necks.
    assert torch.equal(
        fresh.neck.lateral_convs[0].conv.weight,
        source.neck.lateral_convs[0].conv.weight,
    )
    x1 = torch.randn(1, 3, 64, 64)
    x2 = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        assert torch.equal(fresh(x1, x2), source(x1, x2))


def test_load_changeformer_checkpoint_round_trip(configs_dir: Path, tmp_path: Path) -> None:
    """ChangeFormer: siamese MiT backbone + SegFormer head, Open-CD key layout."""
    from opencd_lite import build_model

    config = configs_dir / "changeformer" / "changeformer_mit-b0_256x256_40k_levircd.py"
    source = build_model(config)
    state_dict = {
        **{f"backbone.{k}": v.clone() for k, v in source.backbone.state_dict().items()},
        **{f"decode_head.{k}": v.clone() for k, v in source.decode_head.state_dict().items()},
    }
    path = tmp_path / "changeformer.pth"
    torch.save({"state_dict": state_dict}, path)

    fresh = build_model(config, checkpoint=path)
    assert torch.equal(
        fresh.backbone.layers[0][1][0].attn.attn.in_proj_weight,
        source.backbone.layers[0][1][0].attn.attn.in_proj_weight,
    )
    x1 = torch.randn(1, 3, 64, 64)
    x2 = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        assert torch.equal(fresh(x1, x2), source(x1, x2))


def test_load_changestar_checkpoint_round_trip(configs_dir: Path, tmp_path: Path) -> None:
    """ChangeStar: siamese backbone + parametric FarSegFPN neck + nested head."""
    from opencd_lite import build_model

    config = configs_dir / "changestar" / "changestar_farseg_1x96_256x256_40k_levircd.py"
    source = build_model(config)
    state_dict = {
        **{f"backbone.{k}": v.clone() for k, v in source.backbone.state_dict().items()},
        **{f"neck.{k}": v.clone() for k, v in source.neck.state_dict().items()},
        **{f"decode_head.{k}": v.clone() for k, v in source.decode_head.state_dict().items()},
    }
    path = tmp_path / "changestar.pth"
    torch.save({"state_dict": state_dict}, path)

    fresh = build_model(config, checkpoint=path)
    assert torch.equal(
        fresh.decode_head.seg_head._fsr.scene_encoder[0][0].weight,
        source.decode_head.seg_head._fsr.scene_encoder[0][0].weight,
    )
    x1 = torch.randn(1, 3, 64, 64)
    x2 = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        assert torch.equal(fresh(x1, x2), source(x1, x2))


def test_plain_state_dict_checkpoint(tmp_path: Path) -> None:
    """Checkpoints produced by opencd-lite itself (bare keys) also load."""
    model = CGNet(pretrained=False)
    path = tmp_path / "bare.pth"
    torch.save(model.state_dict(), path)

    fresh = CGNet(pretrained=False)
    report = load_opencd_checkpoint(fresh, path)
    assert not report.missing_keys
    assert not report.unexpected_keys
