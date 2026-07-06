from __future__ import annotations

from pathlib import Path

import yaml

from rl_health_interventions.config.schemas import MDPConfig


def load_config(path: str | Path) -> MDPConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    config = MDPConfig.model_validate(raw)
    if config.transition_model.table_dir is not None:
        resolved = path.parent / config.transition_model.table_dir
        config.transition_model.table_dir = str(resolved)
    return config
