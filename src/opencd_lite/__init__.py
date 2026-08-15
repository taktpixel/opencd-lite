"""opencd-lite: lightweight, mmlab-free Open-CD change detection models.

Public API:

* Models (:class:`CGNet`, :class:`IFN`, ...) — plain ``nn.Module``
  classes, directly instantiable without any config machinery.
* :func:`load_config` — read upstream Open-CD ``.py`` configs.
* :func:`build_model` — assemble a ready-to-use :class:`ChangeDetector`
  from a config, optionally loading a published Open-CD checkpoint.
* :func:`load_opencd_checkpoint` — load Open-CD weights into a model.
* :func:`export_onnx` — export for onnxruntime-only deployment.
* :class:`ONNXChangeDetector` — torch-free inference on an exported
  graph (needs only the ``onnx`` extra: ``numpy`` + ``onnxruntime``).

Attributes are imported lazily (PEP 562): ``import opencd_lite`` and the
torch-free entry points (:class:`ONNXChangeDetector`,
:class:`InferenceConfig`, :class:`PreprocessSpec`, ...) work without
PyTorch installed; the PyTorch model/training/export APIs pull ``torch``
in only when first accessed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "1.3.0"

#: Public attribute -> submodule providing it. Imported on first access
#: so a torch-free install can still use the ONNX inference path.
_LAZY_ATTRS: dict[str, str] = {
    # torch-free entry points
    "ONNXChangeDetector": ".onnx",
    "InferenceConfig": ".protocol",
    "PreprocessSpec": ".transforms",
    "IMAGENET_SPEC": ".transforms",
    "normalize_image_numpy": ".transforms",
    "load_config": ".config",
    "ConfigDict": ".config",
    # PyTorch path (imports torch on access)
    "build_model": ".builder",
    "load_opencd_checkpoint": ".checkpoint",
    "export_onnx": ".export",
    "ChangeDetector": ".inference",
    "normalize_image": ".transforms",
    "CGNet": ".models",
    "IFN": ".models",
    "FC_EF": ".models",
    "FC_Siam_conc": ".models",
    "FC_Siam_diff": ".models",
    "SNUNet_ECAM": ".models",
    "ResNet": ".models",
    "ResNetV1c": ".models",
    "IA_ResNet": ".models",
    "IA_ResNetV1c": ".models",
    "IA_ResNetV1d": ".models",
    "BITHead": ".models",
    "Changer": ".models",
    "STAHead": ".models",
    "DS_FPNHead": ".models",
    "LightCDNet": ".models",
    "TinyFPN": ".models",
    "MixVisionTransformer": ".models",
    "SegformerHead": ".models",
    "TinyNet": ".models",
    "TinyHead": ".models",
    "ChangeStarHead": ".models",
    "FarSegFPN": ".models",
    "FarSegHead": ".models",
    "FeatureFusionNeck": ".models",
    "ConvSegHead": ".models",
    "available_models": ".models",
    "available_heads": ".models",
}

if TYPE_CHECKING:
    from .builder import build_model
    from .checkpoint import load_opencd_checkpoint
    from .config import ConfigDict, load_config
    from .export import export_onnx
    from .inference import ChangeDetector
    from .models import (
        FC_EF,
        IFN,
        BITHead,
        CGNet,
        Changer,
        ChangeStarHead,
        ConvSegHead,
        DS_FPNHead,
        FarSegFPN,
        FarSegHead,
        FC_Siam_conc,
        FC_Siam_diff,
        FeatureFusionNeck,
        IA_ResNet,
        IA_ResNetV1c,
        IA_ResNetV1d,
        LightCDNet,
        MixVisionTransformer,
        ResNet,
        ResNetV1c,
        SegformerHead,
        SNUNet_ECAM,
        STAHead,
        TinyFPN,
        TinyHead,
        TinyNet,
        available_heads,
        available_models,
    )
    from .onnx import ONNXChangeDetector
    from .protocol import InferenceConfig
    from .transforms import (
        IMAGENET_SPEC,
        PreprocessSpec,
        normalize_image,
        normalize_image_numpy,
    )


def __getattr__(name: str) -> Any:
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "BITHead",
    "CGNet",
    "ChangeStarHead",
    "Changer",
    "ConvSegHead",
    "DS_FPNHead",
    "FarSegFPN",
    "FarSegHead",
    "FC_EF",
    "FC_Siam_conc",
    "FC_Siam_diff",
    "FeatureFusionNeck",
    "IA_ResNet",
    "IA_ResNetV1c",
    "IA_ResNetV1d",
    "IFN",
    "LightCDNet",
    "MixVisionTransformer",
    "ResNet",
    "ResNetV1c",
    "SNUNet_ECAM",
    "STAHead",
    "SegformerHead",
    "TinyFPN",
    "TinyHead",
    "TinyNet",
    "ChangeDetector",
    "ConfigDict",
    "IMAGENET_SPEC",
    "InferenceConfig",
    "ONNXChangeDetector",
    "PreprocessSpec",
    "available_heads",
    "available_models",
    "build_model",
    "export_onnx",
    "load_config",
    "load_opencd_checkpoint",
    "normalize_image",
    "normalize_image_numpy",
    "__version__",
]
