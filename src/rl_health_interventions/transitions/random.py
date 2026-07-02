from __future__ import annotations

import numpy as np

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView
from rl_health_interventions.transitions._base import TransitionModel


class RandomTransition(TransitionModel):
    """Transition model with random Dirichlet-generated probabilities.

    Probabilities are lazily generated and cached per (state_key, action, table_type).
    """

    def __init__(self, config: MDPConfig, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._config = config
        self._cache: dict[tuple[str, str, str], tuple[list[str], np.ndarray]] = {}
        self._factor_names: list[str] = list(config.factor_configs.keys())

        # Determine if we are in factored mode (has step_bin and sleep factors)
        self._factored_mode = (
            "step_bin" in config.factor_configs and "sleep" in config.factor_configs
        )
        if self._factored_mode:
            self._step_bin_outcomes = config.factor_configs["step_bin"].names
            self._sleep_outcomes = config.factor_configs["sleep"].names
        else:
            self._step_bin_outcomes = config.factor_configs[self._factor_names[0]].names
            self._sleep_outcomes = []

    def _lazy_cache(
        self, state_key: str, action: str, table_type: str
    ) -> tuple[list[str], np.ndarray]:
        cache_key = (state_key, action, table_type)
        if cache_key not in self._cache:
            if self._factored_mode:
                if table_type == "day_boundary":
                    outcomes = self._sleep_outcomes
                else:
                    outcomes = self._step_bin_outcomes
            else:
                outcomes = self._step_bin_outcomes
            n = len(outcomes)
            probs = self._rng.dirichlet(np.ones(n))
            self._cache[cache_key] = (outcomes, probs)
        return self._cache[cache_key]

    def _make_state_key(self, source: StateView | dict[str, str]) -> str:
        if isinstance(source, StateView):
            return "_".join(getattr(source, name) for name in self._factor_names)
        return "_".join(source[name] for name in self._factor_names)

    def _transition_updates(
        self, state: StateView, action: str, daily_bin: str | None = None
    ) -> dict[str, str]:
        step = state.step_of_day

        if self._factored_mode:
            state_key = self._make_state_key(state)
            if step == 0:
                sleep_outcomes, sleep_probs = self._lazy_cache(
                    state_key, action, "day_boundary"
                )
                sleep_idx = self._rng.choice(len(sleep_outcomes), p=sleep_probs)
                new_sleep = sleep_outcomes[sleep_idx]

                temp_factors = dict(state.factor_values)
                temp_factors["sleep"] = new_sleep
                temp_key = self._make_state_key(temp_factors)
                sb_outcomes, sb_probs = self._lazy_cache(
                    temp_key, action, "within_day_0"
                )
                sb_idx = self._rng.choice(len(sb_outcomes), p=sb_probs)
                new_step_bin = sb_outcomes[sb_idx]

                return {"sleep": new_sleep, "step_bin": new_step_bin}
            else:
                table_type = f"within_day_{step}"
                outcomes, probs = self._lazy_cache(state_key, action, table_type)
                idx = self._rng.choice(len(outcomes), p=probs)
                return {"step_bin": outcomes[idx]}
        else:
            state_key = getattr(state, self._factor_names[0])
            outcomes, probs = self._lazy_cache(state_key, action, "flat")
            idx = self._rng.choice(len(outcomes), p=probs)
            return {self._factor_names[0]: outcomes[idx]}


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY["random"] = RandomTransition
