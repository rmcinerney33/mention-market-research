"""Config loader.

Reads ``config.yaml`` (and optionally ``config.local.yaml`` merged on top).
Kept intentionally minimal — no schema validation library here; downstream
callers should fetch values via :func:`get` and fail fast on missing keys.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config.yaml"
_LOCAL_OVERRIDE_PATH = _REPO_ROOT / "config.local.yaml"


def repo_root() -> Path:
    """Absolute path to the repo root (parent of ``src/``)."""
    return _REPO_ROOT


def load_config(
    path: Path | str | None = None,
    local_override: Path | str | None = _LOCAL_OVERRIDE_PATH,
) -> dict[str, Any]:
    """Load config from YAML, optionally merging a local override on top.

    Parameters
    ----------
    path:
        Path to the base config. Defaults to ``config.yaml`` at repo root.
    local_override:
        Optional path to a local override merged on top. Missing file is OK.
    """
    base_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    with open(base_path) as f:
        cfg = yaml.safe_load(f) or {}

    if local_override is not None:
        override_path = Path(local_override)
        if override_path.exists():
            with open(override_path) as f:
                override = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, override)

    return cfg


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``. Lists are replaced."""
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out
