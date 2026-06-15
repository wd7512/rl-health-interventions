from __future__ import annotations

import numpy as np

from rl_health_interventions.agents._base import Agent
from rl_health_interventions.config.schemas import Action


class EpsilonGreedyAgent(Agent):
    def __init__(
        self,
        n_actions: int = 2,
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.n_actions = n_actions
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)
        self.q_values: dict[Action, float] = {action: 0.0 for action in Action}
        self.counts: dict[Action, int] = {action: 0 for action in Action}

    def select_action(self, state) -> Action:
        if self._rng.random() < self.epsilon:
            actions = list(self.q_values.keys())
            idx = self._rng.integers(len(actions))
            return actions[idx]
        return max(self.q_values, key=lambda a: self.q_values[a])

    def update(self, state, action: Action, reward: float, next_state) -> None:
        self.counts[action] += 1
        n = self.counts[action]
        self.q_values[action] += (reward - self.q_values[action]) / n


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["epsilon_greedy"] = EpsilonGreedyAgent
