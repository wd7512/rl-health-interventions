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
    """Thompson Sampling for binary actions.

    Supports two distribution families for the reward model:

    * ``"beta"`` (default) — Beta-Bernoulli conjugate model.
      ``alpha_prior`` and ``beta_prior`` are the Beta prior parameters
      (default 1.0, 1.0 = uniform).
    * ``"gaussian"`` — Gaussian-Gaussian conjugate model.
      ``alpha_prior`` is initial pseudo-count (default 1.0) and
      ``beta_prior`` is initial sum of rewards (default 0.5, giving
      a prior mean of 0.5 with variance 1.0).

    When ``contextual=True``, maintains separate posterior parameters
    for each ``(context_value, action)`` pair rather than per-action
    globally.
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: int = 42,
        contextual: bool = False,
        context_feature: str | None = None,
        distribution_family: str = "beta",
    ) -> None:
        super().__init__(
            actions=actions,
            seed=seed,
            contextual=contextual,
            context_feature=context_feature,
        )
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.distribution_family = distribution_family
        if distribution_family not in ("beta", "gaussian"):
            raise ValueError(
                f"Unknown distribution_family: {distribution_family}. "
                "Must be 'beta' or 'gaussian'."
            )
        if alpha_prior <= 0.0 or beta_prior <= 0.0:
            raise ValueError("alpha_prior and beta_prior must be strictly positive.")
        self._init_params()

    def _init_params(self) -> None:
        self.posteriors: dict = {}
        if not self.contextual:
            self.posteriors = {
                action: Posterior(alpha=self.alpha_prior, beta=self.beta_prior)
                for action in self._actions
            }

    def _ensure_params(self, key: str | tuple[str, str]) -> None:
        if key not in self.posteriors:
            self.posteriors[key] = Posterior(
                alpha=self.alpha_prior, beta=self.beta_prior
            )

    def _gaussian_posterior(self, n: float, sum_rewards: float) -> tuple[float, float]:
        mu_0 = 0.5
        tau2_0 = 1.0
        sigma2 = 0.25
        tau2_n = 1.0 / (1.0 / tau2_0 + n / sigma2)
        mu_n = tau2_n * (mu_0 / tau2_0 + sum_rewards / sigma2)
        return mu_n, tau2_n

    def select_action(self, state) -> str:
        samples = {}
        for action in self._actions:
            key = self._get_context_key(state, action)
            self._ensure_params(key)
            posterior = self.posteriors[key]
            if self.distribution_family == "beta":
                samples[action] = float(self._rng.beta(posterior.alpha, posterior.beta))
            else:
                mu_n, var_n = self._gaussian_posterior(posterior.alpha, posterior.beta)
                samples[action] = float(
                    np.clip(self._rng.normal(mu_n, np.sqrt(var_n)), 0.0, 1.0)
                )
        return max(samples, key=lambda a: samples[a])

    def update(self, state, action: str, reward: float, next_state) -> None:
        key = self._get_context_key(state, action)
        self._ensure_params(key)
        p = self.posteriors[key]
        if self.distribution_family == "beta":
            if reward > 0.0:
                self.posteriors[key] = Posterior(alpha=p.alpha + 1, beta=p.beta)
            else:
                self.posteriors[key] = Posterior(alpha=p.alpha, beta=p.beta + 1)
        else:
            self.posteriors[key] = Posterior(alpha=p.alpha + 1.0, beta=p.beta + reward)
