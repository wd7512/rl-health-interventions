from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typing_extensions import override

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView
from rl_health_interventions.transitions._base import TransitionModel

_PROB_TOLERANCE = 1e-6


class BootstrapTransition(TransitionModel):
    """Transition model that samples from precomputed bootstrap tables.

    At day boundaries (step_of_day == 0), sleep is sampled from a
    day_boundary table keyed by (step_bin_daily, burden, day_of_week, sleep).
    Within each day, step_bin is sampled from within_day tables keyed by
    (step_bin, burden, action, day_of_week, sleep).
    """

    def __init__(self, config: MDPConfig, seed: int = 42) -> None:
        super().__init__(config, seed=seed)
        self._rng = np.random.default_rng(seed)
        self._table_dir = config.transition_model.table_dir
        self._tables: dict[str, dict] = {}
        self._load_tables()

    def _validate_and_store(self, filename: str, table_data: dict) -> None:
        """Validate probabilities sum to 1.0, then store the table."""
        for key, probs in table_data.items():
            total = sum(probs.values())
            if abs(total - 1.0) > _PROB_TOLERANCE:
                msg = (
                    f"Probabilities for key '{key}' in {filename} "
                    f"sum to {total}, expected 1.0"
                )
                raise ValueError(msg)
        table_name = filename.replace(".json", "")
        self._tables[table_name] = table_data

    def _load_tables(self) -> None:
        if self._table_dir is None:
            msg = "BootstrapTransition requires table_dir in config.transition_model"
            raise ValueError(msg)

        table_dir = Path(self._table_dir)
        filenames = ["day_boundary.json"] + [f"within_day_{i}.json" for i in range(5)]

        for filename in filenames:
            filepath = table_dir / filename
            if not filepath.exists():
                msg = (
                    f"Missing bootstrap table file: {filepath}. "
                    f"Expected {filename} in {table_dir}"
                )
                raise FileNotFoundError(msg)
            with filepath.open(encoding="utf-8") as f:
                table_data: dict[str, dict[str, float]] = json.load(f)
            self._validate_and_store(filename, table_data)

    def _build_key(
        self, state: StateView, action: str, *, for_day_boundary: bool = False
    ) -> str:
        if for_day_boundary:
            return (
                f"{state.step_bin_daily},{state.burden},"
                f"{state.day_of_week},{state.sleep}"
            )
        return (
            f"{state.step_bin},{state.burden},{action},"
            f"{state.day_of_week},{state.sleep}"
        )

    def _sample(self, table: dict[str, dict[str, float]], key: str) -> str:
        probs = table[key]
        outcomes = list(probs.keys())
        probabilities = list(probs.values())
        return str(self._rng.choice(outcomes, p=probabilities))

    @override
    def transition(self, state: StateView, action: str) -> dict[str, str]:
        updates: dict[str, str] = {}

        if state.step_of_day == 0:
            if not hasattr(state, "step_bin_daily"):
                msg = (
                    "step_bin_daily not found on state at step_of_day=0. "
                    "The environment should inject step_bin_daily before "
                    "calling transition."
                )
                raise ValueError(msg)

            db_key = self._build_key(state, action, for_day_boundary=True)
            sleep_next = self._sample(self._tables["day_boundary"], db_key)
            updates["sleep"] = sleep_next
            state = state.with_factors(sleep=sleep_next)

        wd_key = self._build_key(state, action)
        table_idx = min(state.step_of_day, 4)
        step_bin_next = self._sample(self._tables[f"within_day_{table_idx}"], wd_key)
        updates["step_bin"] = step_bin_next

        return updates


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY.register("bootstrap", BootstrapTransition)
