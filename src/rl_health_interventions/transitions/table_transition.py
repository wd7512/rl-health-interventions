from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import numpy as np
from typing_extensions import override

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView
from rl_health_interventions.transitions._base import TransitionModel

logger = logging.getLogger(__name__)


class TableTransition(TransitionModel):
    """Transition model loaded from pre-computed JSON tables.

    Supports two table formats:

    **Flat format** (sprint1): keys are ``{fv1}|{fv2}|...`` or
    ``{fv1}|{fv2}|...|{action}``.  Each entry maps to a single distribution.
    The model samples one value and assigns it to one factor (last for
    day_boundary, first for within_day).

    **Per-factor format** (PEARL): JSON has ``"_format": "per_factor"`` marker.
    Keys are ``{factor_name: {factor_value: {"_": {target: prob}}}}`` for
    day_boundary, or ``{factor_name: {"{fv}|{action}": {"_": {...}}}}`` for
    within_day.  The model samples each stochastic factor independently.
    """

    def __init__(self, config: MDPConfig, seed: int = 42) -> None:
        super().__init__(config, seed=seed)
        self._rng = np.random.default_rng(seed)
        self._per_factor = False
        # Flat format storage
        self._day_boundary: dict[str, tuple[list[str], np.ndarray]] = {}
        self._within_day: list[dict[str, tuple[list[str], np.ndarray]]] = []
        self._db_factor_names: list[str] = []
        self._wd_factor_names: list[str] = []
        self._wd_action_idx: int = -1
        # Per-factor format storage
        self._pf_db: dict[str, dict[str, tuple[list[str], np.ndarray]]] = {}
        self._pf_wd: list[dict[str, dict[str, tuple[list[str], np.ndarray]]]] = []
        self._load_tables()

    @property
    def day_boundary(self) -> dict[str, tuple[list[str], np.ndarray]]:
        if self._per_factor:
            return self._flatten_pf_db()
        return self._day_boundary

    @property
    def within_day(self) -> list[dict[str, tuple[list[str], np.ndarray]]]:
        if self._per_factor:
            return [self._flatten_pf_wd_step(s) for s in self._pf_wd]
        return self._within_day

    def _flatten_pf_db(self) -> dict[str, tuple[list[str], np.ndarray]]:
        """Flatten per-factor day_boundary for Bayesian P-success compatibility."""
        result: dict[str, tuple[list[str], np.ndarray]] = {}
        stochastic = self._stochastic_factors
        factor_value_lists = []
        for name in stochastic:
            var_cfg = self._config.state.variables.get(name)
            if var_cfg is None:
                continue
            factor_value_lists.append(var_cfg.names)
        for combo in itertools.product(*factor_value_lists):
            key = "|".join(combo)
            # Use first factor's distribution for the combo (same as original)
            first_name = stochastic[0]
            first_val = combo[0]
            if first_name in self._pf_db and first_val in self._pf_db[first_name]:
                result[key] = self._pf_db[first_name][first_val]
        return result

    def _flatten_pf_wd_step(  # noqa: C901
        self, step: dict[str, dict[str, tuple[list[str], np.ndarray]]]
    ) -> dict[str, tuple[list[str], np.ndarray]]:
        """Flatten per-factor within_day for one step."""
        import itertools as _it

        result: dict[str, tuple[list[str], np.ndarray]] = {}
        stochastic = self._stochastic_factors
        actions = list(self._config.actions.keys())
        factor_value_lists = []
        for name in stochastic:
            var_cfg = self._config.state.variables.get(name)
            if var_cfg is None:
                continue
            factor_value_lists.append(var_cfg.names)
        for combo in _it.product(*factor_value_lists):
            for action in actions:
                full_key = "|".join(combo) + "|" + action
                for name, fv in zip(stochastic, combo, strict=True):
                    sa_key = f"{fv}|{action}"
                    if name in step and sa_key in step[name]:
                        result[full_key] = step[name][sa_key]
                        break
        return result

    def _load_tables(self) -> None:
        table_dir_str = self._config.transition_model.table_dir
        if table_dir_str is None:
            msg = "table_dir is required for table_transition"
            raise ValueError(msg)
        table_dir = Path(table_dir_str)

        # Load day_boundary
        db_path = table_dir / "day_boundary.json"
        with db_path.open(encoding="utf-8") as f:
            db_data = json.load(f)

        if db_data.get("_format") == "per_factor":
            self._per_factor = True
            self._load_per_factor(db_data, table_dir)
        else:
            self._per_factor = False
            self._load_flat(db_data, table_dir)

    def _load_per_factor(  # noqa: C901, PLR0912
        self,
        db_data: dict,
        table_dir: Path,
    ) -> None:
        """Load per-factor format tables."""
        # day_boundary: {factor_name: {fv: {"_": {target: prob}}}}
        for factor_name, fv_table in db_data.items():
            if factor_name.startswith("_"):
                continue
            self._pf_db[factor_name] = {}
            for fv, dist in fv_table.items():
                self._pf_db[factor_name][fv] = self._parse_single_dist(dist)

        # within_day: {factor_name: {"fv|action": {"_": {target: prob}}}}
        self._pf_wd = []
        for i in range(self._config.steps_per_day):
            wd_path = table_dir / f"within_day_{i}.json"
            with wd_path.open(encoding="utf-8") as f:
                wd_data = json.load(f)
            step_tables: dict[str, dict[str, tuple[list[str], np.ndarray]]] = {}
            for factor_name, fv_table in wd_data.items():
                if factor_name.startswith("_"):
                    continue
                step_tables[factor_name] = {}
                for key, dist in fv_table.items():
                    step_tables[factor_name][key] = self._parse_single_dist(dist)
            self._pf_wd.append(step_tables)

    def _load_flat(self, db_data: dict, table_dir: Path) -> None:  # noqa: C901
        """Load flat (sprint1) format tables."""
        self._day_boundary = self._parse_flat_table(db_data)
        self._within_day = []
        for i in range(self._config.steps_per_day):
            wd_path = table_dir / f"within_day_{i}.json"
            with wd_path.open(encoding="utf-8") as f:
                wd_data = json.load(f)
            self._within_day.append(self._parse_flat_table(wd_data))

        # Infer factor name order from table keys
        first_db_key = ""
        if self._day_boundary:
            first_db_key = next(iter(self._day_boundary))
            self._db_factor_names = self._infer_factor_order(first_db_key)
        if self._within_day and first_db_key:
            first_wd_key = next(iter(self._within_day[0]))
            db_parts = first_db_key.split("|")
            wd_parts = first_wd_key.split("|")
            self._wd_action_idx = len(wd_parts)
            for i, (dp, wp) in enumerate(zip(db_parts, wd_parts, strict=False)):
                if dp != wp:
                    self._wd_action_idx = i
                    break
            factor_parts = [
                wp for j, wp in enumerate(wd_parts) if j != self._wd_action_idx
            ]
            self._wd_factor_names = self._infer_factor_order("|".join(factor_parts))

    def _parse_single_dist(self, dist: dict) -> tuple[list[str], np.ndarray]:
        """Parse a single {"_": {target: prob}} distribution."""
        if "_" not in dist:
            msg = f"Missing '_' key in distribution: {dist}"
            raise ValueError(msg)
        outcomes = dist["_"]
        if not outcomes:
            msg = f"Empty outcomes in distribution: {dist}"
            raise ValueError(msg)
        targets = list(outcomes.keys())
        probs = np.array(list(outcomes.values()), dtype=np.float64)
        total = probs.sum()
        if total <= 0 or not np.isfinite(total):
            msg = f"Invalid probability sum: {dist}"
            raise ValueError(msg)
        probs /= total
        return (targets, probs)

    def _parse_flat_table(self, data: dict) -> dict[str, tuple[list[str], np.ndarray]]:
        result: dict[str, tuple[list[str], np.ndarray]] = {}
        for key, actions in data.items():
            if key.startswith("_"):
                continue
            if "_" not in actions:
                msg = f"Missing '_' key in table entry: {key}"
                raise ValueError(msg)
            result[key] = self._parse_single_dist(actions)
        return result

    def _infer_factor_order(self, table_key: str) -> list[str]:
        """Infer factor name order from a table key."""
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

    def _sample_dist(self, dist: tuple[list[str], np.ndarray]) -> str:
        targets, probs = dist
        idx = self._rng.choice(len(targets), p=probs)
        return str(targets[idx])

    @override
    def transition(self, state: StateView, action: str) -> dict[str, str]:
        if self._per_factor:
            return self._transition_per_factor(state, action)
        return self._transition_flat(state, action)

    def _transition_per_factor(  # noqa: C901, PLR0912
        self, state: StateView, action: str
    ) -> dict[str, str]:
        """Sample each stochastic factor independently (PEARL format)."""
        updates: dict[str, str] = {}
        stochastic = self._stochastic_factors

        # day_boundary: sample each factor independently
        if state.step_of_day == 0:
            for name in stochastic:
                if name not in self._pf_db:
                    continue
                current_val = getattr(state, name)
                if current_val in self._pf_db[name]:
                    updates[name] = self._sample_dist(self._pf_db[name][current_val])
                else:
                    logger.warning(
                        "Missing day_boundary key for factor %s value %s",
                        name,
                        current_val,
                    )

        if state.step_of_day >= len(self._pf_wd):
            msg = (
                f"step_of_day {state.step_of_day} exceeds within_day table count "
                f"{len(self._pf_wd)}"
            )
            raise IndexError(msg)

        # within_day: sample each stochastic factor independently
        step_tables = self._pf_wd[state.step_of_day]
        for name in stochastic:
            if name not in step_tables:
                continue
            current_val = getattr(state, name)
            sa_key = f"{current_val}|{action}"
            if sa_key in step_tables[name]:
                updates[name] = self._sample_dist(step_tables[name][sa_key])
            else:
                logger.warning(
                    "Missing within_day_%d key for %s=%s, action=%s",
                    state.step_of_day,
                    name,
                    current_val,
                    action,
                )
        return updates

    def _transition_flat(  # noqa: C901, PLR0912
        self, state: StateView, action: str
    ) -> dict[str, str]:
        """Flat format: sample one value per table entry (sprint1 format)."""
        updates: dict[str, str] = {}
        if state.step_of_day == 0:
            db_key = self._build_state_key(state, action, for_within_day=False)
            if db_key in self._day_boundary:
                sampled = self._sample(self._day_boundary, db_key)
                if len(self._db_factor_names) == len(self._wd_factor_names):
                    name = self._db_factor_names[-1]
                    updates[name] = sampled
                else:
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
                name = self._wd_factor_names[0]
                updates[name] = sampled
            else:
                for name in self._wd_factor_names:
                    if name in state.factor_values and name not in updates:
                        updates[name] = sampled
        else:
            logger.warning("Missing within_day_%d key: %s", state.step_of_day, wd_key)
        return updates


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY.register("table_transition", TableTransition)
