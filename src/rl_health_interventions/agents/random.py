from __future__ import annotations

import random

from rl_health_interventions.agents._base import Agent
from rl_health_interventions.config.schemas import Action


class RandomAgent(Agent):
    """Baseline: uniform random action selection. No learning."""

    def __init__(self, n_actions: int = 2, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._actions = list(Action)[:n_actions]

    def select_action(self, state) -> Action:
        return self._rng.choice(self._actions)

    def update(self, state, action: Action, reward: float, next_state) -> None:
        pass  # no learning


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["random"] = RandomAgent
