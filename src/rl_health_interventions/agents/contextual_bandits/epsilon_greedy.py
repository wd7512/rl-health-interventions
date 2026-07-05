from __future__ import annotations

from typing_extensions import override

from rl_health_interventions.agents.contextual_bandits._base import (
    ContextualBanditAgent,
)


class EpsilonGreedyAgent(ContextualBanditAgent):
    """Epsilon-greedy action selection with incremental Q-learning.

    When ``contextual=True``, maintains separate ``(q_value, count)``
    entries for each ``(context_value, action)`` pair rather than
    per-action globally.
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        epsilon: float = 0.1,
        seed: int = 42,
        contextual: bool = False,
        context_feature: str | None = None,
    ) -> None:
        super().__init__(
            actions=actions,
            seed=seed,
            contextual=contextual,
            context_feature=context_feature,
        )
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be between 0.0 and 1.0 inclusive.")
        self.epsilon = epsilon
        self._init_params()

    def _init_params(self) -> None:
        self.q_values: dict = {}
        self.counts: dict = {}
        if not self.contextual:
            self.q_values = dict.fromkeys(self._actions, 0.0)
            self.counts = dict.fromkeys(self._actions, 0)

    def _ensure_params(self, key: str | tuple[str, str]) -> None:
        if key not in self.q_values:
            self.q_values[key] = 0.0
            self.counts[key] = 0

    @override
    def select_action(self, state) -> str:
        if self._rng.random() < self.epsilon:
            idx = self._rng.integers(len(self._actions))
            return self._actions[idx]
        action_values = {}
        for action in self._actions:
            key = self._get_context_key(state, action)
            self._ensure_params(key)
            action_values[action] = self.q_values[key]
        max_q = max(action_values.values())
        best = [a for a, q in action_values.items() if q == max_q]
        idx = self._rng.integers(len(best))
        return best[idx]

    @override
    def update(self, state, action: str, reward: float, next_state) -> None:
        key = self._get_context_key(state, action)
        self._ensure_params(key)
        self.counts[key] += 1
        n = self.counts[key]
        self.q_values[key] += (reward - self.q_values[key]) / n
