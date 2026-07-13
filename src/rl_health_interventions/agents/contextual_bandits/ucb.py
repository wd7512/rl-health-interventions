from __future__ import annotations

import math

import numpy as np

from typing_extensions import override

from rl_health_interventions.agents.contextual_bandits._base import (
    ContextualBanditAgent,
    _NumpyDictWrapper,
)


class UCBAgent(ContextualBanditAgent):
    """Upper Confidence Bound (UCB1) action selection.

    When ``contextual=True``, maintains separate ``(q_value, count)``
    entries for each ``(context_value, action)`` pair rather than
    per-action globally.  The exploration bonus uses per-context
    totals (each context is an independent bandit problem).

    For non-contextual mode, uses NumPy arrays internally for Q-values and counts
    for better performance, but provides dict-like access for backward compatibility.
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        c: float = 2.0,
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
        if c <= 0.0:
            raise ValueError("c must be strictly positive.")
        self.c = c
        self._total_steps = 0
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
        if self._use_numpy:
            # Vectorized operations for non-contextual mode
            for action_idx in range(self._n_actions):
                if self.counts_array[action_idx] == 0:
                    return self._actions[action_idx]

            total = self._total_steps
            # Vectorized UCB calculation
            exploration = self.c * np.sqrt(
                np.log(total) / np.maximum(self.counts_array, 1)
            )
            ucb_values = self.q_values_array + exploration
            max_ucb = np.max(ucb_values)
            best_indices = np.where(ucb_values == max_ucb)[0]
            idx = self._rng.integers(len(best_indices))
            return self._actions[best_indices[idx]]
        else:
            # Dict-based for contextual mode
            for action in self._actions:
                key = self._get_context_key(state, action)
                self._ensure_params(key)
                if self.counts[key] == 0:
                    return action

            if self.contextual:
                total = sum(
                    self.counts[self._get_context_key(state, a)] for a in self._actions
                )
            else:
                total = self._total_steps
            ucb_values = {}
            for action in self._actions:
                key = self._get_context_key(state, action)
                n_a = self.counts[key]
                exploration = self.c * math.sqrt(math.log(total) / n_a)
                ucb_values[action] = self.q_values[key] + exploration

            max_ucb = max(ucb_values.values())
            best = [a for a, v in ucb_values.items() if v == max_ucb]
            idx = self._rng.integers(len(best))
            return best[idx]

    @override
    def update(self, state, action: str, reward: float, next_state) -> None:
        self._total_steps += 1

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
