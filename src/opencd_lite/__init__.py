"""opencd-lite: lightweight, mmlab-free Open-CD change detection models.

Public API:

* Models (:class:`CGNet`, :class:`IFN`) — plain ``nn.Module`` classes,
  directly instantiable without any config machinery.
* :func:`load_config` — read upstream Open-CD ``.py`` configs.
* :func:`build_model` — assemble a ready-to-use :class:`ChangeDetector`
  from a config, optionally loading a published Open-CD checkpoint.
* :func:`load_opencd_checkpoint` — load Open-CD weights into a model.
* :func:`export_onnx` — export for onnxruntime-only deployment.
"""

from .builder import build_model
from .checkpoint import load_opencd_checkpoint
from .config import ConfigDict, load_config
from .export import export_onnx
from .inference import ChangeDetector, InferenceConfig
from .models import (
    FC_EF,
    IFN,
    CGNet,
    ConvSegHead,
    FC_Siam_conc,
    FC_Siam_diff,
    SNUNet_ECAM,
    available_models,
)
from .transforms import IMAGENET_SPEC, PreprocessSpec, normalize_image

__version__ = "1.2.0"

__all__ = [
    "CGNet",
    "ConvSegHead",
    "FC_EF",
    "FC_Siam_conc",
    "FC_Siam_diff",
    "IFN",
    "SNUNet_ECAM",
    "ChangeDetector",
    "ConfigDict",
    "IMAGENET_SPEC",
    "InferenceConfig",
    "PreprocessSpec",
    "available_models",
    "build_model",
    "export_onnx",
    "load_config",
    "load_opencd_checkpoint",
    "normalize_image",
    "__version__",
]
