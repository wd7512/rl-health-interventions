from __future__ import annotations

import numpy as np

from rl_health_interventions.config.schemas import (
    ActivityLevel,
    Action,
    MDPConfig,
    TimeOfDay,
)
from rl_health_interventions.transitions._base import TransitionModel


class RuleBasedTransition(TransitionModel):
    def __init__(self, config: MDPConfig, seed: int = 42) -> None:
        self._config = config
        self._rng = np.random.default_rng(seed)
        self._cache: dict[tuple, tuple[list[ActivityLevel], np.ndarray]] = {}
        self._build_cache()

    def _build_cache(self) -> None:
        for state in self._config.activity_levels:
            for action in self._config.actions:
                row = self._config.transition.root[state][action]
                probs = np.array(list(row.values()), dtype=np.float64)
                probs /= probs.sum()
                self._cache[(state, action)] = (list(row.keys()), probs)

    def transition(
        self, state: ActivityLevel, action: Action, time_of_day: TimeOfDay
    ) -> ActivityLevel:
        if self._config.masks.root[time_of_day][state] == 1.0:
            return state
        states, probs = self._cache[(state, action)]
        idx = self._rng.choice(len(states), p=probs)
        return states[idx]


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY["rule_based"] = RuleBasedTransition
