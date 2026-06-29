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
        if not table_dir.is_absolute():
            resolved = Path.cwd() / table_dir
            if resolved.exists():
                table_dir = resolved
        if not table_dir.exists():
            raise FileNotFoundError(
                f"Table directory not found: {table_dir}. "
                f"Use resolve_table_dir(config_path, table_dir) for proper resolution."
            )
        # Detect Sprint1 factored format: need both the specific files and the
        # config factors (step_bin and sleep) that indicate factored mode.
        has_step_bin = "step_bin" in config.factor_configs
        has_sleep = "sleep" in config.factor_configs
        factored_files = [table_dir / f"within_day_{i}.json" for i in range(5)] + [
            table_dir / "day_boundary.json"
        ]
        if has_step_bin and has_sleep and all(f.exists() for f in factored_files):
            self._factored_mode = True
            for table_name in ["day_boundary"] + [f"within_day_{i}" for i in range(5)]:
                file_path = table_dir / f"{table_name}.json"
                data = json.loads(file_path.read_text())
                nested: dict[str, dict[str, tuple[list[str], np.ndarray]]] = {}
                for state, actions in data.items():
                    action_map: dict[str, tuple[list[str], np.ndarray]] = {}
                    for action, probs in actions.items():
                        targets = list(probs.keys())
                        prob_values = np.array(list(probs.values()), dtype=np.float64)
                        prob_values /= prob_values.sum()
                        action_map[action] = (targets, prob_values)
                    nested[state] = action_map
                self._cache[table_name] = nested
        else:
            # MVP flat format: load all JSON files as flat state->action->prob
            table_files = sorted(table_dir.glob("*.json"))
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

    def _make_state_key(self, state: StateView) -> str:
        return "_".join(getattr(state, name) for name in self._factor_names)

    def transition(self, state: StateView, action: str) -> StateView:
        if self._factored_mode:
            step = state.step_of_day
            if step == 0:
                # Step 0: sample sleep from day_boundary, then step_bin from within_day_0
                state_key = self._make_state_key(state)
                sleep_targets, sleep_probs = self._cache["day_boundary"][state_key][
                    action
                ]
                sleep_idx = self._rng.choice(len(sleep_targets), p=sleep_probs)
                new_sleep = sleep_targets[sleep_idx]

                # Update with new sleep, then sample step_bin from within_day_0
                temp_state = state.with_factors(sleep=new_sleep)
                temp_key = self._make_state_key(temp_state)
                sb_targets, sb_probs = self._cache["within_day_0"][temp_key][action]
                sb_idx = self._rng.choice(len(sb_targets), p=sb_probs)
                new_step_bin = sb_targets[sb_idx]

                return state.with_factors(sleep=new_sleep, step_bin=new_step_bin)
            else:
                # Steps 1-4: sample step_bin from within_day_k
                table_key = f"within_day_{step}"
                state_key = self._make_state_key(state)
                targets, probs = self._cache[table_key][state_key][action]
                idx = self._rng.choice(len(targets), p=probs)
                return state.with_factors(step_bin=targets[idx])
        else:
            # Flat mode: single factor
            factor_name = self._factor_names[0]
            current_val = getattr(state, factor_name)
            targets, probs = self._cache[current_val][action]
            idx = self._rng.choice(len(targets), p=probs)
            return state.with_factors(**{factor_name: targets[idx]})


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY["rule_based"] = RuleBasedTransition
