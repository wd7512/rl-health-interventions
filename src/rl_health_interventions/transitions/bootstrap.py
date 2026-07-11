from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typing_extensions import override

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView
from rl_health_interventions.transitions._base import TransitionModel


class BootstrapTransition(TransitionModel):
    def __init__(self, config: MDPConfig, seed: int = 42) -> None:
        super().__init__(config, seed=seed)
        self._rng = np.random.default_rng(seed)
        self._day_boundary: dict[str, tuple[list[str], np.ndarray]] = {}
        self._within_day: list[dict[str, tuple[list[str], np.ndarray]]] = []
        self._load_tables()

    def _load_tables(self) -> None:
        table_dir_str = self._config.transition_model.table_dir
        assert table_dir_str is not None, "table_dir is required for bootstrap"
        table_dir = Path(table_dir_str)
        db_path = table_dir / "day_boundary.json"
        with db_path.open() as f:
            db_data = json.load(f)
        self._day_boundary = self._parse_table(db_data)
        self._within_day = []
        for i in range(self._config.steps_per_day):
            wd_path = table_dir / f"within_day_{i}.json"
            with wd_path.open() as f:
                wd_data = json.load(f)
            self._within_day.append(self._parse_table(wd_data))

    def _parse_table(self, data: dict) -> dict[str, tuple[list[str], np.ndarray]]:
        result: dict[str, tuple[list[str], np.ndarray]] = {}
        for key, actions in data.items():
            outcomes = actions["_"]
            targets = list(outcomes.keys())
            probs = np.array(list(outcomes.values()), dtype=np.float64)
            probs /= probs.sum()
            result[key] = (targets, probs)
        return result

    def _build_state_key(
        self, state: StateView, action: str, *, for_within_day: bool
    ) -> str:
        factors = state.factor_values
        parts = [factors["step_bin"], factors["burden"]]
        if for_within_day:
            parts.append(action)
        parts.extend([factors["day_of_week"], factors["sleep"]])
        return "|".join(parts)

    def _sample(
        self,
        table: dict[str, tuple[list[str], np.ndarray]],
        key: str,
    ) -> str:
        targets, probs = table[key]
        idx = self._rng.choice(len(targets), p=probs)
        return str(targets[idx])

    @override
    def transition(self, state: StateView, action: str) -> dict[str, str]:
        updates: dict[str, str] = {}
        if state.step_of_day == 0:
            db_key = self._build_state_key(state, action, for_within_day=False)
            if db_key in self._day_boundary:
                updates["sleep"] = self._sample(self._day_boundary, db_key)
                state = state.with_factors(sleep=updates["sleep"])
        wd_key = self._build_state_key(state, action, for_within_day=True)
        wd_table = self._within_day[state.step_of_day]
        if wd_key in wd_table:
            updates["step_bin"] = self._sample(wd_table, wd_key)
        return updates


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY.register("bootstrap", BootstrapTransition)
