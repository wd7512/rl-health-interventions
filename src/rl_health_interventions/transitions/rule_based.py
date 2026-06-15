from __future__ import annotations

import numpy as np

from rl_health_interventions.config.schemas import ActivityLevel, Action, MDPConfig, TimeOfDay
from rl_health_interventions.transitions._base import TransitionModel


class RuleBasedTransition(TransitionModel):
    def __init__(self, config: MDPConfig, seed: int = 42) -> None:
        self._config = config
        self._rng = np.random.default_rng(seed)
        self._cache: dict[tuple, list[tuple[ActivityLevel, float]]] = {}
        self._build_cache()

    def _build_cache(self) -> None:
        for state in self._config.activity_levels:
            for action in self._config.actions:
                row = self._config.transition.root[state][action]
                states = list(row.keys())
                probs = np.array([row[s] for s in states], dtype=np.float64)
                self._cache[(state, action)] = list(zip(states, probs.tolist()))

    def transition(
        self, state: ActivityLevel, action: Action, time_of_day: TimeOfDay
    ) -> ActivityLevel:
        if self._config.masks.root[time_of_day][state] == 1.0:
            return state
        states, probs = zip(*self._cache[(state, action)])
        idx = self._rng.choice(len(states), p=np.array(probs))
        return states[idx]


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY["rule_based"] = RuleBasedTransition
