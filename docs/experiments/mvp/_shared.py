"""Shared utilities for MVP experiment scripts — delegates to evaluation package."""

from __future__ import annotations

from pathlib import Path

from rl_health_interventions.evaluation._shared import (
    agent_label,
    run_agent,
)
from rl_health_interventions.evaluation._shared import (
    resolve_config as _resolve_config,
)

_CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def resolve_config(
    name: str | None = None, *, default: str = "mvp_extensions.yaml"
) -> str:
    """Resolve a config name/path for MVP configs."""
    return _resolve_config(_CONFIG_DIR, name, default=default)


__all__ = ["agent_label", "resolve_config", "run_agent"]
