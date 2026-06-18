from __future__ import annotations

import math

from rl_health_interventions.agents.contextual_bandits._base import (
    ContextualBanditAgent,
)


class UCBAgent(ContextualBanditAgent):
    """Upper Confidence Bound (UCB1) action selection.

    When ``contextual=True``, maintains separate ``(q_value, count)``
    entries for each ``(context_value, action)`` pair rather than
    per-action globally.  ``_total_steps`` remains global across all
    contexts.
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        c: float = 2.0,
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
        if c <= 0.0:
            raise ValueError("c must be strictly positive.")
        self.c = c
        self._total_steps = 0
        self._init_params()

    def _init_params(self) -> None:
        self.q_values: dict = {}
        self.counts: dict = {}
        if not self.contextual:
            self.q_values = {a: 0.0 for a in self._actions}
            self.counts = {a: 0 for a in self._actions}

    def _ensure_params(self, key: str | tuple[str, str]) -> None:
        if key not in self.q_values:
            self.q_values[key] = 0.0
            self.counts[key] = 0

    def select_action(self, state) -> str:
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

    def update(self, state, action: str, reward: float, next_state) -> None:
        self._total_steps += 1
        key = self._get_context_key(state, action)
        self._ensure_params(key)
        self.counts[key] += 1
        n = self.counts[key]
        self.q_values[key] += (reward - self.q_values[key]) / n
