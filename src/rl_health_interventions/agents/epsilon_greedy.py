from __future__ import annotations

from rl_health_interventions.agents.contextual_bandits._base import (
    ContextualBanditAgent,
)


class EpsilonGreedyAgent(ContextualBanditAgent):
    """Epsilon-greedy action selection with incremental Q-learning."""

    def __init__(
        self,
        actions: list[str] | None = None,
        epsilon: float = 0.1,
        seed: int = 42,
        contextual: bool = False,
        context_feature: str | None = None,
    ) -> None:
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be between 0.0 and 1.0 inclusive.")
        self.epsilon = epsilon
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

    def _ensure_params(self, key) -> None:
        if key not in self.q_values:
            self.q_values[key] = 0.0
            self.counts[key] = 0

    def select_action(self, state) -> str:
        if self.contextual:
            for action in self._actions:
                key = self._get_context_key(state, action)
                self._ensure_params(key)
            if self._rng.random() < self.epsilon:
                idx = self._rng.integers(len(self._actions))
                return self._actions[idx]
            ctx_q = {
                a: self.q_values[self._get_context_key(state, a)]
                for a in self._actions
            }
            max_q = max(ctx_q.values())
            best = [a for a, q in ctx_q.items() if q == max_q]
            idx = self._rng.integers(len(best))
            return best[idx]
        else:
            if self._rng.random() < self.epsilon:
                idx = self._rng.integers(len(self._actions))
                return self._actions[idx]
            max_q = max(self.q_values.values())
            best = [a for a, q in self.q_values.items() if q == max_q]
            idx = self._rng.integers(len(best))
            return best[idx]

    def update(self, state, action: str, reward: float, next_state) -> None:
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

    REGISTRY["epsilon_greedy"] = EpsilonGreedyAgent
