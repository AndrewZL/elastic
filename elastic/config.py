"""YAML job configs for ELASTIC workers.

Data paths in job YAMLs are relative to ``$ELASTIC_DATA`` (default ``data/``).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
_CONFIGS = ROOT / "configs"
DATA_PATH_KEYS = ("checkpoint", "q_path", "buffer_path", "robomimic_dataset")


def _parse_scalar(raw: str) -> Any:
    return OmegaConf.to_container(OmegaConf.create(f"v: {raw}"))["v"]


def load_yaml(path: Path | str) -> dict:
    cfg = OmegaConf.load(str(path))
    return OmegaConf.to_container(cfg, resolve=True) or {}


def dump_yaml(path: Path | str, cfg: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(dict(cfg)), str(path))


def apply_set(cfg: Mapping[str, Any], items: Optional[Sequence[str]]) -> dict:
    """Apply ``key=value`` overlays (value is YAML)."""
    oc = OmegaConf.create(dict(cfg))
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip()
        parsed = _parse_scalar(raw)
        OmegaConf.update(oc, key, parsed, merge=True)
    return OmegaConf.to_container(oc, resolve=True) or {}


def to_namespace(cfg: Mapping[str, Any], defaults: Mapping[str, Any]) -> SimpleNamespace:
    merged = dict(defaults)
    merged.update(dict(cfg))
    return SimpleNamespace(**merged)


def data_root() -> Path:
    env = os.environ.get("ELASTIC_DATA")
    if env:
        return Path(env)
    return ROOT / "data"


def resolve_data_path(p: Any) -> str:
    if p is None:
        raise ValueError("path is required")
    path = Path(str(p))
    if path.is_absolute():
        return str(path)
    return str(data_root() / path)


def resolve_repo_path(p: Any) -> Path:
    """Resolve paths relative to the elastic repo root (submodules, verifiers, etc.)."""
    if p is None:
        raise ValueError("path is required")
    path = Path(str(p))
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def load_dp_defaults() -> dict:
    return load_yaml(_CONFIGS / "defaults" / "dp.yaml")
