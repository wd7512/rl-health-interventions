from __future__ import annotations

import numpy as np

from rl_health_interventions.agents._base import Agent


class EpsilonGreedyAgent(Agent):
    """Epsilon-greedy action selection with incremental Q-learning."""

    def __init__(
        self,
        actions: list[str] | None = None,
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be between 0.0 and 1.0 inclusive.")
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)
        self._actions = actions or ["nudge", "idle"]
        self.q_values: dict[str, float] = {a: 0.0 for a in self._actions}
        self.counts: dict[str, int] = {a: 0 for a in self._actions}

    def select_action(self, state) -> str:
        if self._rng.random() < self.epsilon:
            idx = self._rng.integers(len(self._actions))
            return self._actions[idx]
        max_q = max(self.q_values.values())
        best = [a for a, q in self.q_values.items() if q == max_q]
        idx = self._rng.integers(len(best))
        return best[idx]

    def update(self, state, action: str, reward: float, next_state) -> None:
        self.counts[action] += 1
        n = self.counts[action]
        self.q_values[action] += (reward - self.q_values[action]) / n


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["epsilon_greedy"] = EpsilonGreedyAgent
