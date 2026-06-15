from __future__ import annotations

import numpy as np

from rl_health_interventions.agents._base import Agent
from rl_health_interventions.config.schemas import Action


class ThompsonSamplingAgent(Agent):
    def __init__(
        self,
        n_actions: int = 2,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.n_actions = n_actions
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self._rng = np.random.default_rng(seed)
        self.posteriors: dict[Action, list[float]] = {
            action: [alpha_prior, beta_prior] for action in Action
        }

    def select_action(self, state) -> Action:
        samples = {
            action: self._rng.beta(posterior[0], posterior[1])
            for action, posterior in self.posteriors.items()
        }
        return max(samples, key=lambda a: samples[a])

    def update(self, state, action: Action, reward: float, next_state) -> None:
        if reward == 1.0:
            self.posteriors[action][0] += 1  # alpha
        else:
            self.posteriors[action][1] += 1  # beta


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["thompson_sampling"] = ThompsonSamplingAgent
