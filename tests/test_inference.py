"""Tests for the ChangeDetector inference wrapper and the config builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from opencd_lite import ChangeDetector, InferenceConfig, build_model
from opencd_lite.transforms import IMAGENET_SPEC, normalize_image, pad_to_divisor


def _random_image(height: int = 96, width: int = 80) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_normalize_image_matches_spec() -> None:
    image = _random_image()
    tensor = normalize_image(image)
    assert tensor.shape == (3, 96, 80)
    expected = (image[..., 0].astype(np.float32) - IMAGENET_SPEC.mean[0]) / IMAGENET_SPEC.std[0]
    np.testing.assert_allclose(tensor[0].numpy(), expected, rtol=1e-5)


def test_pad_to_divisor() -> None:
    batch = torch.ones(1, 3, 100, 66)
    padded, (height, width) = pad_to_divisor(batch, 32)
    assert padded.shape == (1, 3, 128, 96)
    assert (height, width) == (100, 66)
    # Padding is bottom/right with zeros.
    assert padded[..., :100, :66].eq(1).all()
    assert padded[..., 100:, :].eq(0).all()


def test_whole_inference_crops_padding(cgnet_small) -> None:
    detector = ChangeDetector(cgnet_small)
    mask = detector.predict(_random_image(96, 80), _random_image(96, 80))
    assert mask.shape == (96, 80)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 1}


def test_slide_inference(cgnet_small) -> None:
    detector = ChangeDetector(
        cgnet_small,
        inference=InferenceConfig(mode="slide", crop_size=(64, 64), stride=(32, 32), threshold=0.5),
    )
    mask = detector.predict(_random_image(96, 96), _random_image(96, 96))
    assert mask.shape == (96, 96)


def test_slide_falls_back_to_whole_for_small_images(cgnet_small) -> None:
    detector = ChangeDetector(
        cgnet_small,
        inference=InferenceConfig(mode="slide", crop_size=(256, 256), stride=(128, 128)),
    )
    mask = detector.predict(_random_image(64, 64), _random_image(64, 64))
    assert mask.shape == (64, 64)


def test_shape_mismatch_raises(cgnet_small) -> None:
    detector = ChangeDetector(cgnet_small)
    with pytest.raises(ValueError, match="mismatch"):
        detector.predict(_random_image(64, 64), _random_image(96, 96))


def test_invalid_inference_config() -> None:
    with pytest.raises(ValueError, match="mode"):
        InferenceConfig(mode="tile")
    with pytest.raises(ValueError, match="crop_size"):
        InferenceConfig(mode="slide")


def test_build_model_from_cgnet_config(configs_dir: Path) -> None:
    detector = build_model(
        configs_dir / "cgnet" / "cgnet_256x256_40k_levircd.py",
        backbone_overrides={"pretrained": False},
    )
    assert detector.inference_cfg.mode == "slide"
    assert detector.inference_cfg.crop_size == (256, 256)
    assert detector.inference_cfg.threshold == 0.5
    assert detector.preprocess == IMAGENET_SPEC
    assert not detector.training


def test_build_model_from_ifn_config(configs_dir: Path) -> None:
    detector = build_model(
        configs_dir / "ifn" / "ifn_256x256_40k_levircd.py",
        backbone_overrides={"pretrained": False},
    )
    assert detector.inference_cfg.mode == "whole"
    # mmseg default threshold when the config leaves it unset.
    assert detector.inference_cfg.threshold == 0.3

    mask = detector.predict(_random_image(64, 64), _random_image(64, 64))
    assert mask.shape == (64, 64)


def test_build_model_from_fc_siam_configs(configs_dir: Path) -> None:
    """FC-Siam configs use a parametric FCNHead: 2-class logits, argmax."""
    for name in ("fc_ef", "fc_siam_diff", "fc_siam_conc"):
        detector = build_model(configs_dir / "fcsn" / f"{name}_256x256_40k_levircd.py")
        assert detector.decode_head is not None
        assert detector.inference_cfg.out_channels == 2

        mask = detector.predict(_random_image(64, 64), _random_image(64, 64))
        assert mask.shape == (64, 64)
        assert set(np.unique(mask)) <= {0, 1}


def test_build_model_from_snunet_configs(configs_dir: Path) -> None:
    for name, base_channel in (("snunet_c16", 16), ("snunet_c32", 32)):
        detector = build_model(configs_dir / "snunet" / f"{name}_256x256_40k_levircd.py")
        assert detector.decode_head is not None
        assert detector.decode_head.conv_seg.in_channels == base_channel * 4

        mask = detector.predict(_random_image(64, 64), _random_image(64, 64))
        assert mask.shape == (64, 64)


def test_build_model_with_checkpoint(configs_dir: Path, tmp_path: Path, cgnet_small) -> None:
    state_dict = {f"backbone.{k}": v for k, v in cgnet_small.state_dict().items()}
    ckpt = tmp_path / "cgnet.pth"
    torch.save({"state_dict": state_dict}, ckpt)

    detector = build_model(configs_dir / "cgnet" / "cgnet_256x256_40k_levircd.py", checkpoint=ckpt)
    assert torch.equal(
        detector.backbone.state_dict()["conv_reduce_1.conv.weight"],
        cgnet_small.state_dict()["conv_reduce_1.conv.weight"],
    )
