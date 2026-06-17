from __future__ import annotations

import math

import numpy as np

from rl_health_interventions.agents._base import Agent


class UCBAgent(Agent):
    """Upper Confidence Bound (UCB1) action selection."""

    def __init__(
        self,
        actions: list[str] | None = None,
        c: float = 2.0,
        seed: int = 42,
    ) -> None:
        if c <= 0.0:
            raise ValueError("c must be strictly positive.")
        self.c = c
        self._rng = np.random.default_rng(seed)
        self._actions = actions or ["nudge", "idle"]
        self.q_values: dict[str, float] = {a: 0.0 for a in self._actions}
        self.counts: dict[str, int] = {a: 0 for a in self._actions}
        self._total_steps: int = 0

    def select_action(self, state) -> str:
        for action in self._actions:
            if self.counts[action] == 0:
                return action

        total = self._total_steps
        ucb_values = {}
        for action in self._actions:
            n_a = self.counts[action]
            exploration = self.c * math.sqrt(math.log(total) / n_a)
            ucb_values[action] = self.q_values[action] + exploration

        max_ucb = max(ucb_values.values())
        best = [a for a, v in ucb_values.items() if v == max_ucb]
        idx = self._rng.integers(len(best))
        return best[idx]

    def update(self, state, action: str, reward: float, next_state) -> None:
        self._total_steps += 1
        self.counts[action] += 1
        n = self.counts[action]
        self.q_values[action] += (reward - self.q_values[action]) / n


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["ucb"] = UCBAgent
