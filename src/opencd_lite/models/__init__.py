"""Change detection models as plain ``nn.Module`` classes.

Design rule: modules in this package depend only on ``torch`` and
``torchvision``. Training-harness code (Lightning, MLflow, ...) must not
be imported here.
"""

from .bit_head import BITHead
from .cgnet import CGNet
from .changer_head import Changer
from .ds_fpn_head import DS_FPNHead
from .fc_siam import FC_EF, FC_Siam_conc, FC_Siam_diff
from .heads import ConvSegHead
from .ia_resnet import IA_ResNet, IA_ResNetV1c, IA_ResNetV1d
from .ifn import IFN
from .interaction import ChannelExchange, SpatialExchange, TwoIdentity
from .lightcdnet import LightCDNet
from .mit import MixVisionTransformer
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
from .segformer_head import SegformerHead
from .snunet import SNUNet_ECAM
from .sta_head import STAHead
from .tiny_fpn import TinyFPN

__all__ = [
    "BITHead",
    "CGNet",
    "Changer",
    "ChannelExchange",
    "ConvSegHead",
    "DS_FPNHead",
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
    "SpatialExchange",
    "TinyFPN",
    "TwoIdentity",
    "available_heads",
    "available_models",
    "get_head_class",
    "get_model_class",
    "register_head",
    "register_model",
]
