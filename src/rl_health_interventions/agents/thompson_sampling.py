from __future__ import annotations

from typing import NamedTuple

import numpy as np

from rl_health_interventions.agents._base import Agent
from rl_health_interventions.config.schemas import Action


class Posterior(NamedTuple):
    alpha: float
    beta: float


class ThompsonSamplingAgent(Agent):
    """Beta-Bernoulli Thompson Sampling for binary actions.

    NOTE: This is a contextual bandit agent — state is accepted but not
    used in action selection. The posterior is updated globally, not
    per-state. For the MVP (Issue #101) this is correct: the simulator
    uses a stationary transition matrix where E[r|a] is the relevant
    quantity. State-aware agents are planned for Phase 2.
    """

    def __init__(
        self,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        if alpha_prior <= 0.0 or beta_prior <= 0.0:
            raise ValueError("alpha_prior and beta_prior must be strictly positive.")
        self._rng = np.random.default_rng(seed)
        self.posteriors: dict[Action, Posterior] = {
            action: Posterior(alpha=alpha_prior, beta=beta_prior) for action in Action
        }

    def select_action(self, state) -> Action:
        samples = {
            action: self._rng.beta(posterior.alpha, posterior.beta)
            for action, posterior in self.posteriors.items()
        }
        return max(samples, key=lambda a: samples[a])

    def update(self, state, action: Action, reward: float, next_state) -> None:
        p = self.posteriors[action]
        if reward > 0.0:
            self.posteriors[action] = Posterior(alpha=p.alpha + 1, beta=p.beta)
        else:
            self.posteriors[action] = Posterior(alpha=p.alpha, beta=p.beta + 1)


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["thompson_sampling"] = ThompsonSamplingAgent
