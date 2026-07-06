from __future__ import annotations

import numpy as np
from typing_extensions import override

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView
from rl_health_interventions.transitions._base import TransitionModel


class RandomTransition(TransitionModel):
    def __init__(self, config: MDPConfig, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._config = config
        self._cache: dict[tuple[str, ...], tuple[list[str], np.ndarray]] = {}
        self._build_random_cache()

    def _build_random_cache(self) -> None:
        stochastic = self._stochastic_factors
        sleep_names = self._config.state.variables.get("sleep", None)
        step_bin_names = self._config.state.variables.get("step_bin", None)
        if "sleep" in stochastic and sleep_names is not None:
            n = len(sleep_names.names)
            self._cache[("sleep",)] = (
                sleep_names.names,
                self._rng.dirichlet([1.0] * n),
            )
        if "step_bin" in stochastic and step_bin_names is not None:
            n = len(step_bin_names.names)
            self._cache[("step_bin",)] = (
                step_bin_names.names,
                self._rng.dirichlet([1.0] * n),
            )

    @property
    def _stochastic_factors(self) -> list[str]:
        return [
            n for n, c in self._config.state.variables.items() if c.advanced is None
        ]

    @override
    def _sample_sleep(self, state: StateView) -> str:
        targets, probs = self._cache[("sleep",)]
        idx = self._rng.choice(len(targets), p=probs)
        return str(targets[idx])

    @override
    def _sample_step_bin(self, state: StateView, action: str, k: int) -> str:
        targets, probs = self._cache[("step_bin",)]
        idx = self._rng.choice(len(targets), p=probs)
        return str(targets[idx])

    @override
    def transition(self, state: StateView, action: str) -> dict[str, str]:
        updates: dict[str, str] = {}
        if state.step_of_day == 0:
            updates["sleep"] = self._sample_sleep(state)
            state = state.with_factors(sleep=updates["sleep"])
        updates["step_bin"] = self._sample_step_bin(state, action, k=state.step_of_day)
        return updates


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY.register("random", RandomTransition)
