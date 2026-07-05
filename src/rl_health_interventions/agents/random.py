from __future__ import annotations

import numpy as np
from typing_extensions import override

from rl_health_interventions.agents._base import Agent


class RandomAgent(Agent):
    """Baseline: uniform random action selection. No learning."""

    def __init__(self, actions: list[str] | None = None, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._actions = actions or ["nudge", "idle"]

    @override
    def select_action(self, state) -> str:
        idx = int(self._rng.integers(len(self._actions)))
        return self._actions[idx]

    @override
    def update(self, state, action: str, reward: float, next_state) -> None:
        pass


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["random"] = RandomAgent
