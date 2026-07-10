"""Loading published Open-CD checkpoints into opencd-lite models.

Open-CD checkpoints are ordinary PyTorch ``state_dict`` files wrapped in
an mmengine envelope::

    {"meta": {...}, "state_dict": {"backbone.<name>": Tensor, ...}, ...}

Open-CD keeps the network under ``backbone.*`` and (for some models) a
learned classifier under ``decode_head.*``. opencd-lite mirrors that key
layout, so mapping reduces to keeping those prefixes (or stripping
``backbone.`` when loading into a bare model) and dropping harness-only
entries.
"""

from __future__ import annotations

import logging
import pickle
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

__all__ = ["LoadReport", "extract_backbone_state_dict", "load_opencd_checkpoint"]

logger = logging.getLogger(__name__)

_BACKBONE_PREFIX = "backbone."
#: Key prefixes that belong to the mmlab training harness, not the model.
_IGNORED_PREFIXES = ("decode_head.", "auxiliary_head.", "neck.", "data_preprocessor.")


@dataclass
class LoadReport:
    """Outcome of a checkpoint load."""

    matched_keys: list[str] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    ignored_keys: list[str] = field(default_factory=list)

    def raise_on_mismatch(self) -> None:
        if self.missing_keys or self.unexpected_keys:
            raise RuntimeError(
                "Checkpoint does not match the model.\n"
                f"  missing keys: {self.missing_keys}\n"
                f"  unexpected keys: {self.unexpected_keys}"
            )


class _StubbingUnpickler(pickle.Unpickler):
    """Unpickler that replaces unresolvable classes with inert stubs.

    Real Open-CD checkpoints pickle mmengine bookkeeping objects (e.g.
    ``mmengine.logging.history_buffer.HistoryBuffer`` inside the message
    hub). opencd-lite deliberately does not depend on mmengine, so those
    classes cannot be imported; they are also irrelevant — only the
    ``state_dict`` tensors matter. Any class that cannot be resolved is
    substituted with a do-nothing placeholder so unpickling can proceed.
    """

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            logger.debug("Stubbing unresolvable pickle global %s.%s", module, name)
            return _make_stub(module, name)


class _StubMeta(type):
    """Metaclass letting stub classes tolerate arbitrary attribute access.

    Checkpoints may pickle references to class attributes (e.g.
    ``getattr(HistoryBuffer, "min")`` for registered statistics methods);
    resolving them on a stub must not fail.
    """

    def __getattr__(cls, name: str) -> type:
        return _make_stub(cls.__module__, f"{cls.__qualname__}.{name}")


def _make_stub(module: str, name: str) -> type:
    return _StubMeta(
        name,
        (),
        {
            "__module__": module,
            "__init__": lambda self, *args, **kwargs: None,
            "__setstate__": lambda self, state: None,
            "__getattr__": lambda self, name: None,
        },
    )


#: Module accepted by ``torch.load(pickle_module=...)`` (a real ModuleType,
#: since torch inspects its ``__name__``).
_STUB_PICKLE_MODULE = types.ModuleType("opencd_lite._stub_pickle")
_STUB_PICKLE_MODULE.Unpickler = _StubbingUnpickler  # type: ignore[attr-defined]
_STUB_PICKLE_MODULE.load = pickle.load  # type: ignore[attr-defined]


def _read_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read a checkpoint file, tolerating mmengine metadata in the pickle."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - fall back for mmengine envelopes
        warnings.warn(
            f"{path}: not loadable with weights_only=True; falling back to full "
            "unpickling (unresolvable harness classes are stubbed out). "
            "Only load checkpoints from sources you trust.",
            stacklevel=3,
        )
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            pickle_module=_STUB_PICKLE_MODULE,
        )


def extract_backbone_state_dict(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Extract model weights from an Open-CD checkpoint dict.

    Args:
        checkpoint: A loaded checkpoint, either the mmengine envelope or a
            raw ``state_dict``.

    Returns:
        A ``(state_dict, ignored_keys)`` tuple where ``state_dict`` keys
        have the ``backbone.`` prefix removed and ``ignored_keys`` lists
        harness-only entries that were dropped.
    """
    state_dict = checkpoint.get("state_dict", checkpoint)
    extracted: dict[str, torch.Tensor] = {}
    ignored: list[str] = []
    for key, value in state_dict.items():
        if key.startswith(_BACKBONE_PREFIX):
            extracted[key.removeprefix(_BACKBONE_PREFIX)] = value
        elif key.startswith(_IGNORED_PREFIXES):
            ignored.append(key)
        else:
            # No recognized prefix: assume the checkpoint stores bare
            # model keys already (e.g. one produced by opencd-lite).
            extracted[key] = value
    return extracted, ignored


def _extract_detector_state_dict(
    checkpoint: dict[str, Any], *, with_decode_head: bool
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Extract detector-level weights (``backbone.*`` and optionally ``decode_head.*``).

    Keys keep their prefixes so they can be loaded into a
    :class:`~opencd_lite.inference.ChangeDetector` directly; bare keys
    (checkpoints holding a plain model ``state_dict``) are mapped under
    ``backbone.``.
    """
    state_dict = checkpoint.get("state_dict", checkpoint)
    extracted: dict[str, torch.Tensor] = {}
    ignored: list[str] = []
    for key, value in state_dict.items():
        if key.startswith(_BACKBONE_PREFIX) or key.startswith("decode_head.") and with_decode_head:
            extracted[key] = value
        elif key.startswith(_IGNORED_PREFIXES):
            ignored.append(key)
        else:
            extracted[f"{_BACKBONE_PREFIX}{key}"] = value
    return extracted, ignored


def load_opencd_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
) -> LoadReport:
    """Load an Open-CD (or opencd-lite) checkpoint into a model.

    Args:
        model: Either a bare network (e.g. ``CGNet``) or a
            :class:`~opencd_lite.inference.ChangeDetector` wrapper. For a
            wrapper, ``backbone`` receives the model weights and — when a
            parametric ``decode_head`` is present — ``decode_head.*``
            checkpoint keys are loaded into it as well.
        path: Path to the ``.pth`` checkpoint file.
        strict: Raise if any model key is missing or any checkpoint model
            key is unexpected. Harness-only keys are always ignored.

    Returns:
        A :class:`LoadReport` describing matched/ignored keys.
    """
    checkpoint = _read_checkpoint(path)

    if _is_bare_model(model):
        state_dict, ignored = extract_backbone_state_dict(checkpoint)
        result = model.load_state_dict(state_dict, strict=False)
    else:
        has_head = isinstance(getattr(model, "decode_head", None), nn.Module)
        state_dict, ignored = _extract_detector_state_dict(checkpoint, with_decode_head=has_head)
        result = model.load_state_dict(state_dict, strict=False)

    report = LoadReport(
        matched_keys=[k for k in state_dict if k not in result.unexpected_keys],
        missing_keys=list(result.missing_keys),
        unexpected_keys=list(result.unexpected_keys),
        ignored_keys=ignored,
    )
    if strict:
        report.raise_on_mismatch()
    if ignored:
        logger.debug("Ignored %d harness-only checkpoint keys", len(ignored))
    return report


def _is_bare_model(model: nn.Module) -> bool:
    """Return True unless the module wraps its network in a ``backbone`` attribute."""
    backbone = getattr(model, "backbone", None)
    return not isinstance(backbone, nn.Module)
