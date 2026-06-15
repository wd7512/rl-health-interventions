from __future__ import annotations

import numpy as np

from rl_health_interventions.agents._base import Agent
from rl_health_interventions.config.schemas import Action


class RandomAgent(Agent):
    """Baseline: uniform random action selection. No learning."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._actions = list(Action)

    def select_action(self, state) -> Action:
        idx = int(self._rng.integers(len(self._actions)))
        return self._actions[idx]

    def update(self, state, action: Action, reward: float, next_state) -> None:
        pass


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["random"] = RandomAgent
