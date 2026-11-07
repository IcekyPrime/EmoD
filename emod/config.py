"""YAML configuration loading with recursive `_base_` inheritance.

All user-facing paths in configs are relative to the project root. Resolution
happens at runtime through `resolve_path`, so the same config works on any
machine as long as the directory layout is preserved.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_MISSING = object()
_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


class ConfigError(RuntimeError):
    pass


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge mappings; lists and scalars from `override` win."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load(path: Path, seen: set) -> Dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"cyclic _base_ reference at {path}")
    seen = seen | {path}

    raw = _read_yaml(path)
    bases: Iterable[str] = raw.pop("_base_", []) or []
    merged: Dict[str, Any] = {}
    for base in bases:
        base_path = Path(base)
        if not base_path.is_absolute():
            candidate = path.parent / base_path
            if not candidate.exists():
                # Fall back to configs/ so nested _base_ files can be referenced
                # by their top-level relative name.
                candidate = PROJECT_ROOT / "configs" / base_path
            base_path = candidate
        merged = deep_merge(merged, _load(base_path, seen))
    return deep_merge(merged, raw)


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if _NUM_RE.match(text):
        return float(text) if any(ch in text.lower() for ch in (".", "e")) else int(text)
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Apply `a.b.c=value` overrides. Values are parsed as YAML scalars."""
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override must look like key=value, got: {item}")
        dotted, value = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = out
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = _parse_scalar(value.strip())
    return out


def load_config(
    path: str | Path,
    extras: Optional[Iterable[str | Path]] = None,
    overrides: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Load a stage config, then merge optional extras (e.g. a GPU profile)."""
    cfg = _load(Path(path), set())
    for extra in extras or []:
        if extra is None or str(extra) == "":
            continue
        cfg = deep_merge(cfg, _load(Path(extra), set()))
    return apply_overrides(cfg, overrides)


def get(cfg: Dict[str, Any], dotted: str, default: Any = _MISSING) -> Any:
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            if default is _MISSING:
                raise ConfigError(f"missing config key: {dotted}")
            return default
        node = node[key]
    return node


def resolve_path(value: str | Path) -> Path:
    """Resolve a config path against the project root (absolute paths pass through)."""
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def as_rel(path: str | Path) -> str:
    """Best-effort project-relative string, used when writing manifests/logs."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_phone_set(path: str | Path) -> List[str]:
    data = _read_yaml(resolve_path(path))
    phones = data.get("phones")
    if not isinstance(phones, list) or not phones:
        raise ConfigError(f"phone set must contain a non-empty `phones` list: {path}")
    if len(set(phones)) != len(phones):
        raise ConfigError("duplicate entries in phone set")
    return [str(p) for p in phones]


def num_processes(cfg: Dict[str, Any]) -> int:
    """Resolve `runtime.num_processes` to an integer in [1, 8]."""
    import torch

    value = get(cfg, "runtime.num_processes", "auto")
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    visible = max(1, min(visible, 8))
    if value in ("auto", "AUTO", None):
        return visible
    return max(1, min(int(value), 8))
