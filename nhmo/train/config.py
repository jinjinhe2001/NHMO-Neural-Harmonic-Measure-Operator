"""YAML config loader with `inherit:` + env-var interpolation.

Promoted from Phase 4's `tests/test_config.py::_load_yaml_with_inherit`
to a real module. Used by `nhmo/train/trainer.py::main()` and any future
Phase 6/7 entry points.

Supports:
  - `inherit: ../default.yaml` — single-level include; parent merged with
    child overrides (deep merge).
  - `${VAR:-fallback}` — env-var substitution with fallback.
  - `${VAR}` — env-var substitution; raises KeyError if unset and no
    fallback.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml


_ENV_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _substitute_env(value):
    """Recursively substitute env-var placeholders in strings."""
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    if isinstance(value, str):
        def replace(match):
            var, fallback = match.group(1), match.group(2)
            if var in os.environ:
                return os.environ[var]
            if fallback is not None:
                return fallback
            raise KeyError(
                f"Config references env var {var!r} which is unset and has "
                f"no fallback. Set {var}= or add a default in the config."
            )
        return _ENV_VAR_RE.sub(replace, value)
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config, resolving `inherit:` and env vars."""
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    if "inherit" in cfg:
        inherit = cfg.pop("inherit")
        parent_path = (path.parent / inherit).resolve()
        parent = load_config(parent_path)
        cfg = _deep_merge(parent, cfg)
    return _substitute_env(cfg)
