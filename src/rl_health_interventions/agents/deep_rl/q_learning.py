from __future__ import annotations

import numpy as np
from typing_extensions import override

from rl_health_interventions.agents._base import Agent
from rl_health_interventions.agents.deep_rl._base import (
    state_to_key,
    validate_gamma,
    validate_lr,
    validate_unit_interval,
)


class QLearningAgent(Agent):
    def __init__(
        self,
        actions: list[str] | None = None,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        validate_lr(lr)
        validate_gamma(gamma)
        validate_unit_interval(epsilon, "epsilon")
        self._rng = np.random.default_rng(seed)
        self._actions = actions or ["nudge", "idle"]
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: dict[tuple[str, ...], np.ndarray] = {}

    def _values_for(self, state) -> np.ndarray:
        key = state_to_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(len(self._actions), dtype=np.float64)
        return self.q_table[key]

    @override
    def select_action(self, state) -> str:
        q_values = self._values_for(state)
        if self._rng.random() < self.epsilon:
            return self._actions[int(self._rng.integers(len(self._actions)))]
        best = np.flatnonzero(q_values == np.max(q_values))
        idx = int(self._rng.choice(best))
        return self._actions[idx]

    @override
    def update(
        self, state, action: str, reward: float, next_state, done: bool = False
    ) -> None:
        q_values = self._values_for(state)
        action_idx = self._actions.index(action)
        if done:
            td_target = reward
        else:
            next_q = self._values_for(next_state)
            td_target = reward + self.gamma * float(np.max(next_q))
        q_values[action_idx] += self.lr * (td_target - q_values[action_idx])
