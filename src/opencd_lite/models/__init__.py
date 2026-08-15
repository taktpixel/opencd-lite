"""Change detection models as plain ``nn.Module`` classes.

Design rule: modules in this package depend only on ``torch`` and
``torchvision``. Training-harness code (Lightning, MLflow, ...) must not
be imported here.
"""

from .bit_head import BITHead
from .cgnet import CGNet
from .fc_siam import FC_EF, FC_Siam_conc, FC_Siam_diff
from .heads import ConvSegHead
from .ifn import IFN
from .necks import FeatureFusionNeck
from .registry import (
    available_heads,
    available_models,
    get_head_class,
    get_model_class,
    register_head,
    register_model,
)
from .resnet import ResNet, ResNetV1c
from .snunet import SNUNet_ECAM

__all__ = [
    "BITHead",
    "CGNet",
    "ConvSegHead",
    "FC_EF",
    "FC_Siam_conc",
    "FC_Siam_diff",
    "FeatureFusionNeck",
    "IFN",
    "ResNet",
    "ResNetV1c",
    "SNUNet_ECAM",
    "available_heads",
    "available_models",
    "get_head_class",
    "get_model_class",
    "register_head",
    "register_model",
]
