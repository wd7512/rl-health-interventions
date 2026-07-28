from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from typing_extensions import override

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView
from rl_health_interventions.transitions._base import TransitionModel
from rl_health_interventions.transitions._table_loader import TableLoader
from rl_health_interventions.transitions._table_validator import TableValidator

logger = logging.getLogger(__name__)


class TableTransition(TransitionModel):
    def __init__(
        self,
        config: MDPConfig,
        seed: int = 42,
        loader: TableLoader | None = None,
        validator: TableValidator | None = None,
    ) -> None:
        super().__init__(config, seed=seed)
        self._rng = np.random.default_rng(seed)
        self._lookup: dict[str, dict[str, tuple[list[str], np.ndarray]]] = {}
        self._include_step_of_day = False
        self._loader = loader or TableLoader()
        self._validator = validator or TableValidator()
        self._load_tables()

    # ── Table loading & validation ────────────────────────────────────────

    def _load_tables(self) -> None:
        table_dir = self._resolve_table_dir()
        tables = self._loader.load_tables(table_dir)

        errors: list[str] = []
        valid_tables: list[dict] = []

        for data in tables:
            try:
                self._validator.validate(data, self._config)
                valid_tables.append(data)
            except ValueError as exc:
                errors.append(str(exc))

        if errors:
            raise ValueError("; ".join(errors))

        for data in valid_tables:
            self._process_file(data)

    def _resolve_table_dir(self) -> Path:
        table_dir_str = self._config.transition_model.table_dir
        if table_dir_str is None:
            msg = "table_dir is required for table transition"
            raise ValueError(msg)
        table_dir = Path(table_dir_str)
        if not table_dir.is_dir():
            msg = f"table_dir not found: {table_dir}"
            raise FileNotFoundError(msg)
        return table_dir

    # ── File processing ───────────────────────────────────────────────────

    def _process_file(self, data: dict) -> None:
        global_state = data.get("global_state", {})
        if not isinstance(global_state, dict):
            global_state = {}
        if "step_of_day" in global_state:
            self._include_step_of_day = True

        config_var_order = list(self._config.state.variables.keys())

        for entry in data["transitions"]:
            entry_state = entry["state"]
            config_vars = set(self._config.state.variables)
            if not set(entry_state.keys()).issubset(config_vars):
                continue

            key = self._build_entry_key(
                entry_state, global_state, entry["action"], config_var_order
            )
            converted = self._convert_probabilities(entry["next_state_probs"])
            self._lookup[key] = converted

    def _build_entry_key(
        self, state: dict, global_state: dict, action: str, var_order: list[str]
    ) -> str:
        merged = {**state, **global_state}
        parts = [str(merged.get(f, "")) for f in var_order]
        if self._include_step_of_day:
            parts.append(str(merged.get("step_of_day", 0)))
        parts.append(action)
        return "|".join(parts)

    @staticmethod
    def _convert_probabilities(
        next_state_probs: dict,
    ) -> dict[str, tuple[list[str], np.ndarray]]:
        converted: dict[str, tuple[list[str], np.ndarray]] = {}
        for factor, probs in next_state_probs.items():
            targets = list(probs.keys())
            prob_values = np.array(list(probs.values()), dtype=np.float64)
            converted[factor] = (targets, prob_values)
        return converted

    # ── State key building ────────────────────────────────────────────────

    def _build_state_key(self, state: StateView, action: str) -> str:
        factor_values = state.factor_values
        config_var_order = list(self._config.state.variables.keys())
        parts = [str(factor_values[f]) for f in config_var_order]
        if self._include_step_of_day:
            parts.append(str(state.step_of_day))
        parts.append(action)
        return "|".join(parts)

    # ── Transition ────────────────────────────────────────────────────────

    @override
    def transition(self, state: StateView, action: str) -> dict[str, str]:
        state_key = self._build_state_key(state, action)
        if state_key not in self._lookup:
            logger.warning("Missing state-action pair: %s", state_key)
            return {}

        factor_dists = self._lookup[state_key]
        updates: dict[str, str] = {}
        for factor in self._stochastic_factors:
            if factor in factor_dists:
                targets, probs = factor_dists[factor]
                idx = self._rng.choice(len(targets), p=probs)
                updates[factor] = str(targets[idx])
        return updates


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY.register("table", TableTransition)
