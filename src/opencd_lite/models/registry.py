"""Minimal model registry mapping Open-CD config ``type`` names to classes.

This replaces the mmengine Registry with a plain dictionary: models are
ordinary ``nn.Module`` classes that can always be instantiated directly;
the registry only serves config-driven construction
(:func:`opencd_lite.builder.build_model`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from torch import nn

_MODELS: dict[str, type[nn.Module]] = {}

_T = TypeVar("_T", bound=type[nn.Module])


def register_model(name: str) -> Callable[[_T], _T]:
    """Class decorator registering a model under its Open-CD ``type`` name."""

    def decorator(cls: _T) -> _T:
        if name in _MODELS:
            raise ValueError(f"Model {name!r} is already registered")
        _MODELS[name] = cls
        return cls

    return decorator


def get_model_class(name: str) -> type[nn.Module]:
    """Look up a registered model class by its Open-CD ``type`` name."""
    try:
        return _MODELS[name]
    except KeyError:
        supported = ", ".join(sorted(_MODELS))
        raise KeyError(f"Unknown model type {name!r}. Supported types: {supported}") from None


def available_models() -> list[str]:
    """Return the sorted list of registered model type names."""
    return sorted(_MODELS)
