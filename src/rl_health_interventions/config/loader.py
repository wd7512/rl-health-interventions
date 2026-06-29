from __future__ import annotations

from pathlib import Path

import yaml

from rl_health_interventions.config.schemas import MDPConfig


def resolve_table_dir(config_path: Path, table_dir: str) -> Path:
    return config_path.parent / table_dir


def load_config(path: str | Path) -> MDPConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return MDPConfig.model_validate(raw)
