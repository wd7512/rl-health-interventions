from __future__ import annotations

import math

from rl_health_interventions.agents.contextual_bandits._base import (
    ContextualBanditAgent,
)


class UCBAgent(ContextualBanditAgent):
    """Upper Confidence Bound (UCB1) action selection."""

    def __init__(
        self,
        actions: list[str] | None = None,
        c: float = 2.0,
        seed: int = 42,
        contextual: bool = False,
        context_feature: str | None = None,
    ) -> None:
        if c <= 0.0:
            raise ValueError("c must be strictly positive.")
        self.c = c
        super().__init__(
            actions=actions,
            seed=seed,
            contextual=contextual,
            context_feature=context_feature,
        )
        if contextual:
            self.q_values: dict = {}
            self.counts: dict = {}
        else:
            self.q_values = {a: 0.0 for a in self._actions}
            self.counts = {a: 0 for a in self._actions}
        self._total_steps: int = 0

    def _ensure_params(self, key) -> None:
        if key not in self.q_values:
            self.q_values[key] = 0.0
            self.counts[key] = 0

    def select_action(self, state) -> str:
        if self.contextual:
            for action in self._actions:
                key = self._get_context_key(state, action)
                self._ensure_params(key)
                if self.counts[key] == 0:
                    return action
            total = self._total_steps
            ucb_values = {}
            for action in self._actions:
                key = self._get_context_key(state, action)
                n_a = self.counts[key]
                exploration = self.c * math.sqrt(math.log(total) / n_a)
                ucb_values[action] = self.q_values[key] + exploration
        else:
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
        if self.contextual:
            key = self._get_context_key(state, action)
            self._ensure_params(key)
            self.counts[key] += 1
            n = self.counts[key]
            self.q_values[key] += (reward - self.q_values[key]) / n
        else:
            self.counts[action] += 1
            n = self.counts[action]
            self.q_values[action] += (reward - self.q_values[action]) / n


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["ucb"] = UCBAgent
