from __future__ import annotations

from typing import NamedTuple

from typing_extensions import override

from rl_health_interventions.agents.contextual_bandits._base import (
    ContextualBanditAgent,
)


class Posterior(NamedTuple):
    alpha: float
    beta: float


class ThompsonSamplingAgent(ContextualBanditAgent):
    """Beta-Bernoulli Thompson Sampling for binary actions.

    When ``contextual=True``, maintains separate ``(alpha, beta)``
    posteriors for each ``(context_value, action)`` pair rather than
    per-action globally.
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: int = 42,
        contextual: bool = False,
        context_features: str | None = None,
    ) -> None:
        super().__init__(
            actions=actions,
            seed=seed,
            contextual=contextual,
            context_features=context_features,
        )
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
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

    def _ensure_params(self, key: str | tuple[str, ...]) -> None:
        if key not in self.posteriors:
            self.posteriors[key] = Posterior(
                alpha=self.alpha_prior, beta=self.beta_prior
            )

    @override
    def select_action(self, state) -> str:
        samples = {}
        for action in self._actions:
            key = self._get_context_key(state, action)
            self._ensure_params(key)
            posterior = self.posteriors[key]
            samples[action] = float(self._rng.beta(posterior.alpha, posterior.beta))
        return max(samples, key=lambda a: samples[a])

    @override
    def update(self, state, action: str, reward: float, next_state) -> None:
        key = self._get_context_key(state, action)
        self._ensure_params(key)
        p = self.posteriors[key]
        if reward > 0.0:
            self.posteriors[key] = Posterior(alpha=p.alpha + 1, beta=p.beta)
        else:
            self.posteriors[key] = Posterior(alpha=p.alpha, beta=p.beta + 1)
