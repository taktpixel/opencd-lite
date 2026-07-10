"""Change detection models as plain ``nn.Module`` classes.

Design rule: modules in this package depend only on ``torch`` and
``torchvision``. Training-harness code (Lightning, MLflow, ...) must not
be imported here.
"""

from .cgnet import CGNet
from .fc_siam import FC_EF, FC_Siam_conc, FC_Siam_diff
from .heads import ConvSegHead
from .ifn import IFN
from .registry import available_models, get_model_class, register_model
from .snunet import SNUNet_ECAM

__all__ = [
    "CGNet",
    "ConvSegHead",
    "FC_EF",
    "FC_Siam_conc",
    "FC_Siam_diff",
    "IFN",
    "SNUNet_ECAM",
    "available_models",
    "get_model_class",
    "register_model",
]
