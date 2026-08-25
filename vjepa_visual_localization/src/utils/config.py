"""YAML configuration loading with explicit validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject an empty or non-mapping document."""

    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a required mapping section."""

    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"configuration section '{name}' is missing or invalid")
    return value
