from __future__ import annotations

from typing import NamedTuple

import numpy as np

from rl_health_interventions.agents._base import Agent


class Posterior(NamedTuple):
    alpha: float
    beta: float


class ThompsonSamplingAgent(Agent):
    """Beta-Bernoulli Thompson Sampling for binary actions.

    When *contextual* is ``True`` the agent maintains separate
    ``(alpha, beta)`` posteriors for each ``(context_value, action)``
    pair, keyed on ``state.<context_feature>``.
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: int = 42,
        contextual: bool = False,
        context_feature: str | None = None,
    ) -> None:
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        if alpha_prior <= 0.0 or beta_prior <= 0.0:
            raise ValueError("alpha_prior and beta_prior must be strictly positive.")
        self._rng = np.random.default_rng(seed)
        self._actions = actions or ["nudge", "idle"]
        self.contextual = contextual
        self.context_feature = context_feature
        self._init_posteriors()

    def _init_posteriors(self) -> None:
        if self.contextual:
            self.posteriors: dict[str | tuple[str, str], Posterior] = {}
        else:
            self.posteriors = {
                action: Posterior(alpha=self.alpha_prior, beta=self.beta_prior)
                for action in self._actions
            }

    def _get_context_key(self, state, action: str) -> tuple[str, str] | str:
        """Extract the posterior key from state, or raise on bad input."""
        if not self.contextual:
            return action
        ctx_attr = self.context_feature
        if ctx_attr is None:
            raise ValueError("context_feature must be set when contextual=True")
        if state is None:
            raise ValueError("state cannot be None for contextual agent")
        context_value = getattr(state, ctx_attr, None)
        if context_value is None:
            raise ValueError(f"state is missing required context feature '{ctx_attr}'")
        return (context_value, action)

    def _ensure_posterior(self, key: tuple[str, str] | str) -> Posterior:
        if key not in self.posteriors:
            self.posteriors[key] = Posterior(
                alpha=self.alpha_prior, beta=self.beta_prior
            )
        return self.posteriors[key]

    def select_action(self, state) -> str:
        samples: dict[str, float] = {}
        for action in self._actions:
            key = self._get_context_key(state, action)
            p = self._ensure_posterior(key)
            samples[action] = float(self._rng.beta(p.alpha, p.beta))
        return max(samples, key=lambda a: samples[a])

    def update(self, state, action: str, reward: float, next_state) -> None:
        key = self._get_context_key(state, action)
        p = self._ensure_posterior(key)
        if reward > 0.0:
            self.posteriors[key] = Posterior(alpha=p.alpha + 1, beta=p.beta)
        else:
            self.posteriors[key] = Posterior(alpha=p.alpha, beta=p.beta + 1)


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["thompson_sampling"] = ThompsonSamplingAgent
