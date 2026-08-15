"""Build ready-to-use models from Open-CD config files.

This is the bridge between upstream Open-CD configs and opencd-lite:
:func:`build_model` reads the ``model`` section of a config (loaded with
:func:`opencd_lite.config.load_config`) and assembles a
:class:`~opencd_lite.inference.ChangeDetector` with the matching
preprocessing and test-time protocol. Training-related sections of the
config (optimizer, dataloaders, hooks, ...) are intentionally ignored.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch import nn

from .checkpoint import load_opencd_checkpoint
from .config import load_config
from .inference import ChangeDetector, InferenceConfig
from .models import ConvSegHead, FeatureFusionNeck, get_head_class, get_model_class
from .transforms import IMAGENET_SPEC, PreprocessSpec

__all__ = ["IDENTITY_HEAD_TYPES", "build_model"]

#: ``DIEncoderDecoder`` feeds both images to the backbone at once;
#: ``SiamEncoderDecoder`` runs the shared backbone per image and fuses
#: the two feature pyramids with a (parameter-free) neck.
_SUPPORTED_DETECTOR_TYPES = ("DIEncoderDecoder", "SiamEncoderDecoder")
#: Parameter-free Open-CD heads: the model output already is the prediction.
IDENTITY_HEAD_TYPES = ("IdentityHead", "DSIdentityHead")
#: Config keys of registered decode heads that belong to the training
#: harness (or are handled by the inference wrapper), not the module.
_HEAD_HARNESS_KEYS = ("type", "loss_decode", "sampler", "ignore_index", "in_index")
#: mmseg's BaseDecodeHead default when a binary head leaves threshold unset.
_DEFAULT_BINARY_THRESHOLD = 0.3
#: mmseg's BaseDecodeHead default dropout before the classifier.
_DEFAULT_HEAD_DROPOUT = 0.1


def build_model(
    config: str | Path | Mapping[str, Any],
    checkpoint: str | Path | None = None,
    *,
    backbone_overrides: Mapping[str, Any] | None = None,
) -> ChangeDetector:
    """Build a :class:`ChangeDetector` from an Open-CD config.

    Args:
        config: Path to an Open-CD ``.py`` config file, or an already
            loaded config mapping.
        checkpoint: Optional path to an Open-CD checkpoint to load. When
            given, ImageNet pretraining of the encoder is skipped (the
            checkpoint overwrites it anyway).
        backbone_overrides: Optional constructor-argument overrides for
            the model (e.g. ``{"pretrained": False}``).

    Returns:
        A ``ChangeDetector`` in eval mode.
    """
    cfg = load_config(config) if isinstance(config, (str, Path)) else config
    model_cfg = cfg.get("model")
    if model_cfg is None:
        raise KeyError("Config has no 'model' section")

    detector_type = model_cfg.get("type")
    if detector_type not in _SUPPORTED_DETECTOR_TYPES:
        raise NotImplementedError(
            f"Detector type {detector_type!r} is not supported yet "
            f"(supported: {', '.join(_SUPPORTED_DETECTOR_TYPES)})"
        )
    siamese = detector_type == "SiamEncoderDecoder"
    if not siamese and isinstance(model_cfg.get("neck"), Mapping):
        raise NotImplementedError(
            "DIEncoderDecoder configs with a 'neck' section are not supported"
        )

    backbone_cfg = dict(model_cfg["backbone"])
    model_type = backbone_cfg.pop("type")
    model_class = get_model_class(model_type)
    if checkpoint is not None and _accepts_kwarg(model_class, "pretrained"):
        # The checkpoint overwrites every weight; skip the (large) download
        # of ImageNet encoder weights even when the config requests them.
        backbone_cfg["pretrained"] = False
    if backbone_overrides:
        backbone_cfg.update(backbone_overrides)
    model = model_class(**backbone_cfg)

    detector = ChangeDetector(
        model,
        preprocess=_build_preprocess_spec(model_cfg.get("data_preprocessor")),
        inference=_build_inference_config(model_cfg),
        decode_head=_build_decode_head(model_cfg.get("decode_head", {})),
        neck=_build_neck(model_cfg.get("neck")) if siamese else None,
        siamese=siamese,
    )
    if checkpoint is not None:
        load_opencd_checkpoint(detector, checkpoint)
    detector.eval()
    return detector


def _build_decode_head(head_cfg: Mapping[str, Any]) -> nn.Module | None:
    """Build the parametric decode head, or ``None`` for identity heads."""
    head_type = head_cfg.get("type", "IdentityHead")
    if head_type in IDENTITY_HEAD_TYPES:
        return None
    if head_type == "mmseg.FCNHead":
        # Only the degenerate FCNHead used by Open-CD configs (a pure
        # dropout + 1x1-conv classifier) is supported.
        if head_cfg.get("num_convs", 2) != 0 or head_cfg.get("concat_input", True):
            raise NotImplementedError(
                "mmseg.FCNHead is only supported with num_convs=0 and concat_input=False"
            )
        return ConvSegHead(
            in_channels=head_cfg["channels"],
            num_classes=head_cfg.get("out_channels", head_cfg["num_classes"]),
            dropout_ratio=head_cfg.get("dropout_ratio", _DEFAULT_HEAD_DROPOUT),
        )
    try:
        head_class = get_head_class(head_type)
    except KeyError:
        raise NotImplementedError(f"Decode head type {head_type!r} is not supported yet") from None
    head_kwargs = {k: v for k, v in head_cfg.items() if k not in _HEAD_HARNESS_KEYS}
    return head_class(**head_kwargs)


def _build_neck(neck_cfg: Mapping[str, Any] | None) -> nn.Module:
    """Build the feature-fusion neck of a ``SiamEncoderDecoder`` config."""
    if neck_cfg is None:
        raise NotImplementedError(
            "SiamEncoderDecoder configs without a 'neck' section are not supported yet"
        )
    cfg = dict(neck_cfg)
    neck_type = cfg.pop("type")
    if neck_type != "FeatureFusionNeck":
        raise NotImplementedError(f"Neck type {neck_type!r} is not supported yet")
    return FeatureFusionNeck(**cfg)


def _accepts_kwarg(callable_obj: type, name: str) -> bool:
    """Check whether a class constructor accepts a given keyword argument."""
    return name in inspect.signature(callable_obj).parameters


def _build_preprocess_spec(preprocessor_cfg: Mapping[str, Any] | None) -> PreprocessSpec:
    """Translate an Open-CD ``data_preprocessor`` dict into a :class:`PreprocessSpec`."""
    if preprocessor_cfg is None:
        return IMAGENET_SPEC

    # Open-CD duplicates the 3-channel constants for the concatenated
    # bi-temporal input; both halves are always identical.
    mean = list(preprocessor_cfg.get("mean", IMAGENET_SPEC.mean * 2))
    std = list(preprocessor_cfg.get("std", IMAGENET_SPEC.std * 2))
    if len(mean) == 6:
        if mean[:3] != mean[3:] or std[:3] != std[3:]:
            raise NotImplementedError("Different statistics per temporal image")
        mean, std = mean[:3], std[:3]
    size_divisor = preprocessor_cfg.get("size_divisor", IMAGENET_SPEC.size_divisor)
    return PreprocessSpec(mean=tuple(mean), std=tuple(std), size_divisor=size_divisor)


def _build_inference_config(model_cfg: Mapping[str, Any]) -> InferenceConfig:
    """Translate ``decode_head`` + ``test_cfg`` into an :class:`InferenceConfig`."""
    head_cfg: Mapping[str, Any] = model_cfg.get("decode_head", {})
    out_channels = head_cfg.get("out_channels", head_cfg.get("num_classes", 2))
    threshold = head_cfg.get("threshold")
    if threshold is None:
        threshold = _DEFAULT_BINARY_THRESHOLD

    out_index = head_cfg.get("in_index", -1)
    if isinstance(out_index, (list, tuple)):
        raise NotImplementedError("Multi-input decode heads are not supported yet")

    test_cfg: Mapping[str, Any] = model_cfg.get("test_cfg") or {}
    mode = test_cfg.get("mode", "whole")
    crop_size = test_cfg.get("crop_size")
    stride = test_cfg.get("stride")
    return InferenceConfig(
        mode=mode,
        crop_size=tuple(crop_size) if crop_size is not None else None,
        stride=tuple(stride) if stride is not None else None,
        out_index=out_index,
        out_channels=out_channels,
        threshold=threshold,
    )
