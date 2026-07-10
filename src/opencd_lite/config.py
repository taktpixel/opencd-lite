"""Loader for mmengine-style Python config files, without mmengine.

Open-CD (like all OpenMMLab projects) describes experiments as Python
files containing plain variables, with two special conventions:

* ``_base_``: a path or list of paths (relative to the config file) whose
  contents are loaded first and recursively merged.
* ``_delete_``: when set to ``True`` inside a dict, the dict *replaces*
  the corresponding dict from the base config instead of being merged
  into it.

This module reimplements just enough of those semantics to read the
upstream Open-CD config files unchanged. Config files are executed as
Python, so only load configs from sources you trust.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

__all__ = ["ConfigDict", "load_config"]

_DELETE_KEY = "_delete_"
_BASE_KEY = "_base_"


class ConfigDict(dict):
    """Dict with attribute access, e.g. ``cfg.model.backbone.type``."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None


def _wrap(value: Any) -> Any:
    """Recursively convert plain dicts to :class:`ConfigDict`."""
    if isinstance(value, dict):
        return ConfigDict({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return type(value)(_wrap(v) for v in value)
    return value


def _strip_delete(value: Any) -> Any:
    """Recursively drop ``_delete_`` markers from dicts."""
    if isinstance(value, dict):
        return {k: _strip_delete(v) for k, v in value.items() if k != _DELETE_KEY}
    if isinstance(value, (list, tuple)):
        return type(value)(_strip_delete(v) for v in value)
    return value


def _merge_a_into_b(a: dict, b: dict) -> dict:
    """Merge dict ``a`` (overrides) into dict ``b`` (base), returning a new dict.

    Follows mmengine semantics: nested dicts merge recursively unless the
    override dict carries ``_delete_=True``, in which case it replaces the
    base value entirely. Non-dict values (including lists) always replace.
    """
    merged = dict(b)
    for key, value in a.items():
        if (
            isinstance(value, dict)
            and not value.get(_DELETE_KEY, False)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _merge_a_into_b(value, merged[key])
        else:
            merged[key] = _strip_delete(value)
    return merged


def _exec_config_file(path: Path) -> dict[str, Any]:
    """Execute a config file and return its public module-level variables."""
    source = path.read_text(encoding="utf-8")
    namespace: dict[str, Any] = {"__file__": str(path)}
    code = compile(source, str(path), "exec")
    exec(code, namespace)  # noqa: S102 - documented trust requirement
    return {
        key: value
        for key, value in namespace.items()
        if not key.startswith("__") and not isinstance(value, types.ModuleType)
    }


def _load_config_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg = _exec_config_file(path)
    base_spec = cfg.pop(_BASE_KEY, [])
    if isinstance(base_spec, str):
        base_spec = [base_spec]

    merged_base: dict[str, Any] = {}
    for base_rel in base_spec:
        base_cfg = _load_config_dict((path.parent / base_rel).resolve())
        merged_base = _merge_a_into_b(base_cfg, merged_base)

    return _merge_a_into_b(cfg, merged_base)


def load_config(path: str | Path) -> ConfigDict:
    """Load an (Open-CD compatible) Python config file.

    Args:
        path: Path to a ``.py`` config file. ``_base_`` references are
            resolved relative to the file itself, so upstream Open-CD
            configs can be pointed at in-place.

    Returns:
        The fully merged configuration as a :class:`ConfigDict`.
    """
    return _wrap(_load_config_dict(Path(path).resolve()))
