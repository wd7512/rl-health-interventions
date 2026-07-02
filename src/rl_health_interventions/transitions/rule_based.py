from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView
from rl_health_interventions.transitions._base import TransitionModel


class RuleBasedTransition(TransitionModel):
    def __init__(self, config: MDPConfig, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._config = config
        self._cache: dict[str, Any] = {}
        self._factored_mode = False
        self._factor_names: list[str] = list(config.factor_configs.keys())
        self._load_tables(config)

    def _load_tables(self, config: MDPConfig) -> None:
        if config.transition_model.table_dir is None:
            raise ValueError("rule_based transition requires table_dir in config")
        table_dir = Path(config.transition_model.table_dir)
        if not table_dir.exists():
            raise FileNotFoundError(
                f"Table directory not found: {table_dir}. "
                "Use load_config() which resolves table_dir relative to the config file."
            )
        has_step_bin = "step_bin" in config.factor_configs
        has_sleep = "sleep" in config.factor_configs
        steps = config.steps_per_day
        factored_files = [table_dir / f"within_day_{i}.json" for i in range(steps)] + [
            table_dir / "day_boundary.json"
        ]
        if has_step_bin and has_sleep and all(f.exists() for f in factored_files):
            self._factored_mode = True
            for table_name in ["day_boundary"] + [
                f"within_day_{i}" for i in range(steps)
            ]:
                file_path = table_dir / f"{table_name}.json"
                data = json.loads(file_path.read_text())
                if table_name == "day_boundary":
                    # No action dimension: state → {target: prob}
                    parsed: dict[str, tuple[list[str], np.ndarray]] = {}
                    for state, probs in data.items():
                        targets = list(probs.keys())
                        prob_values = np.array(list(probs.values()), dtype=np.float64)
                        prob_values /= prob_values.sum()
                        parsed[state] = (targets, prob_values)
                    self._cache[table_name] = parsed
                else:
                    nstd: dict[str, dict[str, tuple[list[str], np.ndarray]]] = {}
                    for state, actions in data.items():
                        action_map: dict[str, tuple[list[str], np.ndarray]] = {}
                        for action, probs in actions.items():
                            targets = list(probs.keys())
                            prob_values = np.array(
                                list(probs.values()), dtype=np.float64
                            )
                            prob_values /= prob_values.sum()
                            action_map[action] = (targets, prob_values)
                        nstd[state] = action_map
                    self._cache[table_name] = nstd
        else:
            # MVP flat format: load JSON table file(s).
            if config.transition_model.table is not None:
                table_path = table_dir / config.transition_model.table
                if not table_path.exists():
                    raise FileNotFoundError(f"Table file not found: {table_path}")
                table_files = [table_path]
            else:
                table_files = sorted(table_dir.glob("*.json"))
                # Exclude factored-mode tables (day_boundary, within_day_*)
                table_files = [
                    f
                    for f in table_files
                    if "day_boundary" not in f.name
                    and not f.name.startswith("within_day_")
                ]
                if not table_files:
                    raise FileNotFoundError(f"No JSON table files found in {table_dir}")
            for table_file in table_files:
                data = json.loads(table_file.read_text())
                for state, actions in data.items():
                    action_map: dict[str, tuple[list[str], np.ndarray]] = {}
                    for action, probs in actions.items():
                        targets = list(probs.keys())
                        prob_values = np.array(list(probs.values()), dtype=np.float64)
                        prob_values /= prob_values.sum()
                        action_map[action] = (targets, prob_values)
                    self._cache[state] = action_map

    def _make_state_key(self, source: StateView | dict[str, str]) -> str:
        if isinstance(source, StateView):
            return "_".join(getattr(source, name) for name in self._factor_names)
        return "_".join(source[name] for name in self._factor_names)

    def _transition_updates(
        self, state: StateView, action: str, daily_bin: str | None = None
    ) -> dict[str, str]:
        if self._factored_mode:
            step = state.step_of_day
            if step == 0:
                key_parts = dict(state.factor_values)
                key_parts["step_bin"] = daily_bin or state.step_bin
                state_key = "_".join(key_parts[n] for n in sorted(key_parts))
                sleep_targets, sleep_probs = self._cache["day_boundary"][state_key]
                sleep_idx = self._rng.choice(len(sleep_targets), p=sleep_probs)
                new_sleep = sleep_targets[sleep_idx]

                temp_factors = dict(state.factor_values)
                temp_factors["sleep"] = new_sleep
                temp_key = self._make_state_key(temp_factors)
                sb_targets, sb_probs = self._cache["within_day_0"][temp_key][action]
                sb_idx = self._rng.choice(len(sb_targets), p=sb_probs)
                new_step_bin = sb_targets[sb_idx]

                return {"sleep": new_sleep, "step_bin": new_step_bin}
            else:
                table_key = f"within_day_{step}"
                state_key = self._make_state_key(state)
                targets, probs = self._cache[table_key][state_key][action]
                idx = self._rng.choice(len(targets), p=probs)
                return {"step_bin": targets[idx]}
        else:
            factor_name = self._factor_names[0]
            current_val = getattr(state, factor_name)
            targets, probs = self._cache[current_val][action]
            idx = self._rng.choice(len(targets), p=probs)
            return {factor_name: targets[idx]}


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY["rule_based"] = RuleBasedTransition
