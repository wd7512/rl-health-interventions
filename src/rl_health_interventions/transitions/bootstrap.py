from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from typing_extensions import override

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView
from rl_health_interventions.transitions._base import TransitionModel

logger = logging.getLogger(__name__)


class BootstrapTransition(TransitionModel):
    def __init__(self, config: MDPConfig, seed: int = 42) -> None:
        super().__init__(config, seed=seed)
        self._rng = np.random.default_rng(seed)
        self._day_boundary: dict[str, tuple[list[str], np.ndarray]] = {}
        self._within_day: list[dict[str, tuple[list[str], np.ndarray]]] = []
        self._db_factor_names: list[str] = []
        self._wd_factor_names: list[str] = []
        self._load_tables()

    @property
    def day_boundary(self) -> dict[str, tuple[list[str], np.ndarray]]:
        return self._day_boundary

    @property
    def within_day(self) -> list[dict[str, tuple[list[str], np.ndarray]]]:
        return self._within_day

    def _load_tables(self) -> None:  # noqa: C901
        table_dir_str = self._config.transition_model.table_dir
        if table_dir_str is None:
            msg = "table_dir is required for bootstrap transition"
            raise ValueError(msg)
        table_dir = Path(table_dir_str)
        db_path = table_dir / "day_boundary.json"
        with db_path.open(encoding="utf-8") as f:
            db_data = json.load(f)
        self._day_boundary = self._parse_table(db_data)
        self._within_day = []
        for i in range(self._config.steps_per_day):
            wd_path = table_dir / f"within_day_{i}.json"
            with wd_path.open(encoding="utf-8") as f:
                wd_data = json.load(f)
            self._within_day.append(self._parse_table(wd_data))
        # Infer factor name order from table keys
        self._db_factor_names: list[str] = []
        self._wd_factor_names: list[str] = []
        self._wd_action_idx: int = -1
        first_db_key = ""
        if self._day_boundary:
            first_db_key = next(iter(self._day_boundary))
            self._db_factor_names = self._infer_factor_order(first_db_key)
        if self._within_day and first_db_key:
            first_wd_key = next(iter(self._within_day[0]))
            db_parts = first_db_key.split("|")
            wd_parts = first_wd_key.split("|")
            # Find action position: the part in wd that differs from db at same position
            self._wd_action_idx = len(wd_parts)  # default: action at end
            for i, (dp, wp) in enumerate(zip(db_parts, wd_parts, strict=False)):
                if dp != wp:
                    self._wd_action_idx = i
                    break
            # Infer factor names from within_day key, skipping action position
            factor_parts = [
                wp for j, wp in enumerate(wd_parts) if j != self._wd_action_idx
            ]
            self._wd_factor_names = self._infer_factor_order(
                "|".join(factor_parts)
            )

    def _infer_factor_order(self, table_key: str) -> list[str]:
        """Infer factor name order from a table key.

        Matches values to config variables by checking which variable
        each value belongs to.
        """
        parts = table_key.split("|")
        factor_names: list[str] = []
        used: set[str] = set()
        for val in parts:
            for var_name, var_cfg in self._config.state.variables.items():
                if var_name not in used and val in var_cfg.names:
                    factor_names.append(var_name)
                    used.add(var_name)
                    break
        return factor_names

    def _parse_table(self, data: dict) -> dict[str, tuple[list[str], np.ndarray]]:
        result: dict[str, tuple[list[str], np.ndarray]] = {}
        for key, actions in data.items():
            if "_" not in actions:
                msg = f"Missing '_' key in table entry: {key}"
                raise ValueError(msg)
            outcomes = actions["_"]
            if not outcomes:
                msg = f"Empty outcomes in table entry: {key}"
                raise ValueError(msg)
            targets = list(outcomes.keys())
            probs = np.array(list(outcomes.values()), dtype=np.float64)
            total = probs.sum()
            if total <= 0 or not np.isfinite(total):
                msg = f"Invalid probability sum in table entry: {key}"
                raise ValueError(msg)
            probs /= total
            result[key] = (targets, probs)
        return result

    def _build_state_key(
        self, state: StateView, action: str, *, for_within_day: bool
    ) -> str:
        factors = state.factor_values
        if for_within_day:
            factor_names = self._wd_factor_names
        else:
            factor_names = self._db_factor_names
        parts = [factors[name] for name in factor_names if name in factors]
        if for_within_day:
            parts.insert(self._wd_action_idx, action)
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
    def transition(self, state: StateView, action: str) -> dict[str, str]:  # noqa: C901, PLR0912
        updates: dict[str, str] = {}
        if state.step_of_day == 0:
            db_key = self._build_state_key(state, action, for_within_day=False)
            if db_key in self._day_boundary:
                sampled = self._sample(self._day_boundary, db_key)
                if len(self._db_factor_names) == len(self._wd_factor_names):
                    # Sprint1: day_boundary updates only the last factor
                    name = self._db_factor_names[-1]
                    updates[name] = sampled
                else:
                    # PEARL: day_boundary updates all factors in the key
                    for name in self._db_factor_names:
                        if name in state.factor_values:
                            updates[name] = sampled
                state = state.with_factors(**updates)
            else:
                logger.warning("Missing day_boundary key: %s", db_key)
        if state.step_of_day >= len(self._within_day):
            msg = (
                f"step_of_day {state.step_of_day} exceeds within_day table count "
                f"{len(self._within_day)}"
            )
            raise IndexError(msg)
        wd_key = self._build_state_key(state, action, for_within_day=True)
        wd_table = self._within_day[state.step_of_day]
        if wd_key in wd_table:
            sampled = self._sample(wd_table, wd_key)
            if len(self._db_factor_names) == len(self._wd_factor_names):
                # Sprint1: within_day updates only the first factor
                name = self._wd_factor_names[0]
                updates[name] = sampled
            else:
                # PEARL: within_day updates all factors in the key
                for name in self._wd_factor_names:
                    if name in state.factor_values and name not in updates:
                        updates[name] = sampled
        else:
            logger.warning("Missing within_day_%d key: %s", state.step_of_day, wd_key)
        return updates


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY.register("bootstrap", BootstrapTransition)
