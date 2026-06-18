from __future__ import annotations

from typing import NamedTuple

import numpy as np

from rl_health_interventions.agents.contextual_bandits._base import (
    ContextualBanditAgent,
)


class Posterior(NamedTuple):
    alpha: float
    beta: float


class ThompsonSamplingAgent(ContextualBanditAgent):
    """Beta-Bernoulli Thompson Sampling for binary actions."""

    def __init__(
        self,
        actions: list[str] | None = None,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: int = 42,
        contextual: bool = False,
        context_feature: str | None = None,
    ) -> None:
        if alpha_prior <= 0.0 or beta_prior <= 0.0:
            raise ValueError("alpha_prior and beta_prior must be strictly positive.")
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        super().__init__(
            actions=actions,
            seed=seed,
            contextual=contextual,
            context_feature=context_feature,
        )
        if contextual:
            self.posteriors: dict = {}
        else:
            self.posteriors = {
                action: Posterior(alpha=alpha_prior, beta=beta_prior)
                for action in self._actions
            }

    def _ensure_params(self, key) -> None:
        if key not in self.posteriors:
            self.posteriors[key] = Posterior(
                alpha=self.alpha_prior, beta=self.beta_prior
            )

    def select_action(self, state) -> str:
        if self.contextual:
            for action in self._actions:
                key = self._get_context_key(state, action)
                self._ensure_params(key)
            samples = {
                action: self._rng.beta(
                    self.posteriors[self._get_context_key(state, action)].alpha,
                    self.posteriors[self._get_context_key(state, action)].beta,
                )
                for action in self._actions
            }
        else:
            samples = {
                action: self._rng.beta(posterior.alpha, posterior.beta)
                for action, posterior in self.posteriors.items()
            }
        return max(samples, key=lambda a: samples[a])

    def update(self, state, action: str, reward: float, next_state) -> None:
        if self.contextual:
            key = self._get_context_key(state, action)
            self._ensure_params(key)
            p = self.posteriors[key]
            if reward > 0.0:
                self.posteriors[key] = Posterior(alpha=p.alpha + 1, beta=p.beta)
            else:
                self.posteriors[key] = Posterior(alpha=p.alpha, beta=p.beta + 1)
        else:
            p = self.posteriors[action]
            if reward > 0.0:
                self.posteriors[action] = Posterior(alpha=p.alpha + 1, beta=p.beta)
            else:
                self.posteriors[action] = Posterior(alpha=p.alpha, beta=p.beta + 1)


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["thompson_sampling"] = ThompsonSamplingAgent
