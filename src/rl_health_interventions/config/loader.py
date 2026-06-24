from __future__ import annotations

from pathlib import Path

import yaml

from rl_health_interventions.config.schemas import MDPConfig


def load_config(path: str | Path) -> MDPConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return MDPConfig.model_validate(raw)
