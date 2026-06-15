from __future__ import annotations

import numpy as np

from rl_health_interventions.agents._base import Agent
from rl_health_interventions.config.schemas import Action


class EpsilonGreedyAgent(Agent):
    """Epsilon-greedy action selection with incremental Q-learning.

    NOTE: This is a contextual bandit agent — state is accepted but not
    used in action selection. Q-values are updated globally, not per-state.
    For the MVP (Issue #101) this is correct. State-aware agents are
    planned for Phase 2.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be between 0.0 and 1.0 inclusive.")
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)
        self._actions = list(Action)
        self.q_values: dict[Action, float] = {action: 0.0 for action in Action}
        self.counts: dict[Action, int] = {action: 0 for action in Action}

    def select_action(self, state) -> Action:
        if self._rng.random() < self.epsilon:
            idx = self._rng.integers(len(self._actions))
            return self._actions[idx]
        max_q = max(self.q_values.values())
        best = [a for a, q in self.q_values.items() if q == max_q]
        idx = self._rng.integers(len(best))
        return best[idx]

    def update(self, state, action: Action, reward: float, next_state) -> None:
        self.counts[action] += 1
        n = self.counts[action]
        self.q_values[action] += (reward - self.q_values[action]) / n


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["epsilon_greedy"] = EpsilonGreedyAgent
