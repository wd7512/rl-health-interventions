from __future__ import annotations

import numpy as np

from typing_extensions import override

from rl_health_interventions.agents.contextual_bandits._base import (
    ContextualBanditAgent,
    _NumpyDictWrapper,
)


class EpsilonGreedyAgent(ContextualBanditAgent):
    """Epsilon-greedy action selection with incremental Q-learning.

    When ``contextual=True``, maintains separate ``(q_value, count)``
    entries for each ``(context_value, action)`` pair rather than
    per-action globally.

    For non-contextual mode, uses NumPy arrays internally for Q-values and counts
    for better performance, but provides dict-like access for backward compatibility.
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        epsilon: float = 0.1,
        seed: int = 42,
        contextual: bool = False,
        context_features: str | list[str] | None = None,
    ) -> None:
        super().__init__(
            actions=actions,
            seed=seed,
            contextual=contextual,
            context_features=context_features,
        )
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be between 0.0 and 1.0 inclusive.")
        self.epsilon = epsilon
        self._init_params()

    def _init_params(self) -> None:
        if self._use_numpy:
            # Use NumPy arrays for non-contextual mode
            self.q_values_array: np.ndarray = np.zeros(
                self._n_actions, dtype=np.float64
            )
            self.counts_array: np.ndarray = np.zeros(
                self._n_actions, dtype=np.int64
            )
            # Provide dict-like access for backward compatibility
            self.q_values: _NumpyDictWrapper = _NumpyDictWrapper(
                self.q_values_array, self._action_to_index
            )
            self.counts: _NumpyDictWrapper = _NumpyDictWrapper(
                self.counts_array, self._action_to_index
            )
        else:
            # Use dicts for contextual mode
            self.q_values: dict = {}
            self.counts: dict = {}

    def _ensure_params(self, key: str | tuple[str, ...]) -> None:
        if self._use_numpy:
            return  # NumPy arrays are pre-allocated
        if key not in self.q_values:
            self.q_values[key] = 0.0
            self.counts[key] = 0

    @override
    def select_action(self, state) -> str:
        if self._rng.random() < self.epsilon:
            idx = self._rng.integers(len(self._actions))
            return self._actions[idx]

        if self._use_numpy:
            # Vectorized operations for non-contextual mode
            max_q = np.max(self.q_values_array)
            best_indices = np.where(self.q_values_array == max_q)[0]
            idx = self._rng.integers(len(best_indices))
            return self._actions[best_indices[idx]]
        else:
            # Dict-based for contextual mode
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
        if self._use_numpy:
            # For non-contextual, use action index directly
            action_idx = self._action_to_index[action]
            self.counts_array[action_idx] += 1
            n = self.counts_array[action_idx]
            self.q_values_array[action_idx] += (reward - self.q_values_array[action_idx]) / n
        else:
            # For contextual, use context key
            key = self._get_context_key(state, action)
            self._ensure_params(key)
            self.counts[key] += 1
            n = self.counts[key]
            self.q_values[key] += (reward - self.q_values[key]) / n
