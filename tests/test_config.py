"""Tests for the mmengine-compatible config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from opencd_lite.config import ConfigDict, load_config


def test_load_cgnet_config(configs_dir: Path) -> None:
    cfg = load_config(configs_dir / "cgnet" / "cgnet_256x256_40k_levircd.py")

    # Values from the _base_ model file survive the merge.
    assert cfg.model.type == "DIEncoderDecoder"
    assert cfg.model.backbone.type == "CGNet"
    assert cfg.model.decode_head.threshold == 0.5

    # The leaf config overrides test_cfg via recursive dict merge.
    assert cfg.model.test_cfg.mode == "slide"
    assert tuple(cfg.model.test_cfg.crop_size) == (256, 256)
    assert tuple(cfg.model.test_cfg.stride) == (128, 128)

    # _delete_=True replaces optim_wrapper wholesale and is stripped.
    assert cfg.optim_wrapper.optimizer.lr == 5e-4
    assert "_delete_" not in cfg.optim_wrapper

    # Values from the chained common/_base_ configs are present.
    assert cfg.train_dataloader.dataset.type == "LEVIR_CD_Dataset"
    assert cfg.default_scope == "opencd"


def test_load_ifn_config(configs_dir: Path) -> None:
    cfg = load_config(configs_dir / "ifn" / "ifn_256x256_40k_levircd.py")

    assert cfg.model.backbone.type == "IFN"
    assert cfg.model.test_cfg.mode == "whole"
    # IFN's binary head does not set a threshold in the config.
    assert "threshold" not in cfg.model.decode_head
    # Base optimizer settings are not overridden by the leaf config.
    assert cfg.optim_wrapper.optimizer.lr == 0.001


def test_load_bit_config(configs_dir: Path) -> None:
    cfg = load_config(configs_dir / "bit" / "bit_r18_256x256_40k_levircd.py")

    assert cfg.model.type == "SiamEncoderDecoder"
    assert cfg.model.backbone.type == "mmseg.ResNetV1c"
    assert cfg.model.backbone.num_stages == 3
    assert tuple(cfg.model.backbone.out_indices) == (2,)
    assert cfg.model.neck.type == "FeatureFusionNeck"
    assert cfg.model.neck.policy == "concat"
    assert cfg.model.decode_head.type == "BITHead"
    assert cfg.model.decode_head.channels == 32
    assert cfg.model.test_cfg.mode == "whole"


def test_load_changer_config(configs_dir: Path) -> None:
    cfg = load_config(configs_dir / "changer" / "changer_ex_r18_512x512_40k_levircd.py")

    assert cfg.model.type == "DIEncoderDecoder"
    assert cfg.model.backbone.type == "IA_ResNetV1c"
    # The leaf config overrides the all-identity interaction_cfg.
    assert cfg.model.backbone.interaction_cfg[1]["type"] == "SpatialExchange"
    assert cfg.model.backbone.interaction_cfg[2]["type"] == "ChannelExchange"
    assert cfg.model.decode_head.type == "Changer"
    assert list(cfg.model.decode_head.in_index) == [0, 1, 2, 3]
    assert cfg.model.decode_head.channels == 128


def test_load_stanet_config(configs_dir: Path) -> None:
    cfg = load_config(configs_dir / "stanet" / "stanet_pam_256x256_40k_levircd.py")

    assert cfg.model.type == "SiamEncoderDecoder"
    assert cfg.model.backbone.type == "mmseg.ResNetV1c"
    assert cfg.model.neck.policy == "concat"
    assert cfg.model.decode_head.type == "STAHead"
    # The PAM leaf config overrides the base sa_mode='None'.
    assert cfg.model.decode_head.sa_mode == "PAM"
    assert cfg.model.decode_head.out_channels == 1
    assert cfg.model.decode_head.threshold == 0.5
    assert cfg.model.test_cfg.mode == "slide"


def test_load_lightcdnet_config(configs_dir: Path) -> None:
    cfg = load_config(configs_dir / "lightcdnet" / "lightcdnet_b_256x256_40k_levircd.py")

    assert cfg.model.type == "DIEncoderDecoder"
    assert cfg.model.backbone.type == "LightCDNet"
    # The base-variant leaf overrides net_type and the neck widths.
    assert cfg.model.backbone.net_type == "base"
    assert cfg.model.neck.type == "TinyFPN"
    assert list(cfg.model.neck.in_channels) == [24, 116, 232, 464]
    assert cfg.model.decode_head.type == "DS_FPNHead"
    assert cfg.model.decode_head.dropout_ratio == 0.0


def test_load_changeformer_config(configs_dir: Path) -> None:
    cfg = load_config(configs_dir / "changeformer" / "changeformer_mit-b1_256x256_40k_levircd.py")

    assert cfg.model.type == "SiamEncoderDecoder"
    assert cfg.model.backbone.type == "mmseg.MixVisionTransformer"
    # The b1 leaf overrides the base embed_dims.
    assert cfg.model.backbone.embed_dims == 64
    assert cfg.model.neck.policy == "concat"
    assert cfg.model.decode_head.type == "mmseg.SegformerHead"
    # Doubled in_channels for the concatenated bi-temporal features.
    assert list(cfg.model.decode_head.in_channels) == [128, 256, 640, 1024]


def test_load_changestar_config(configs_dir: Path) -> None:
    cfg = load_config(configs_dir / "changestar" / "changestar_farseg_1x96_512x512_40k_levircd.py")

    assert cfg.model.type == "SiamEncoderDecoder"
    assert cfg.model.backbone.type == "mmseg.ResNetV1c"
    assert cfg.model.neck.type == "FarSegFPN"
    assert cfg.model.neck.out_channels == 256
    assert cfg.model.decode_head.type == "ChangeStarHead"
    assert cfg.model.decode_head.inference_mode == "t1t2"
    assert cfg.model.decode_head.seg_head_cfg.type == "FarSegHead"
    assert cfg.model.decode_head.changemixin_cfg.inner_channels == 96
    assert cfg.model.decode_head.out_channels == 1


def test_load_tinycd_v2_config(configs_dir: Path) -> None:
    cfg = load_config(configs_dir / "tinycd_v2" / "tinycd_v2_s_256x256_40k_levircd.py")

    assert cfg.model.type == "DIEncoderDecoder"
    assert cfg.model.backbone.type == "TinyNet"
    # The small leaf halves the width of the base variant.
    assert cfg.model.backbone.arch == "S"
    assert cfg.model.backbone.widen_factor == 0.5
    assert cfg.model.neck.type == "TinyFPN"
    assert list(cfg.model.neck.in_channels) == [8, 16, 16, 24]
    assert cfg.model.decode_head.type == "TinyHead"
    assert cfg.model.decode_head.priori_attn is True


def test_load_upstream_config_in_place() -> None:
    """The loader must read configs from the original Open-CD checkout."""
    upstream = (
        Path(__file__).resolve().parents[2]
        / "open-cd"
        / "configs"
        / "ifn"
        / "ifn_256x256_40k_levircd.py"
    )
    if not upstream.is_file():
        pytest.skip("upstream open-cd checkout not available")
    cfg = load_config(upstream)
    assert cfg.model.backbone.type == "IFN"


def test_config_dict_attribute_access() -> None:
    cfg = ConfigDict({"a": {"b": 1}})
    assert cfg["a"] == {"b": 1}
    with pytest.raises(AttributeError):
        _ = cfg.missing


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.py")
