"""Nightly batch update that wires together model and proxy updates.

Implements Section 5.4 of the HeartSteps V2 paper.

The nightly update processes a full day's data (5 decision times) and:
    1. Updates the BayesianRewardModel posterior with available observations.
    2. Re-solves the proxy value function and applies the weighted update.
    3. Returns a summary dict with updated parameters.

Reference:
    Liao et al. (2019). "Personalized HeartSteps." arXiv:1909.03539, Section 5.4.
"""

from __future__ import annotations

import logging


from paper_reproduction.heartsteps.bayesian_regression import BayesianRewardModel
from paper_reproduction.heartsteps.proxy_value import ProxyValueFunction

logger = logging.getLogger(__name__)

# TypedDict-like structure for a single observation
Observation = dict


class NightlyUpdater:
    """Coordinates nightly updates of the Bayesian reward model and proxy value.

    Processes daily data and updates the posterior and proxy value function
    after each full day of observations.
    """

    def __init__(
        self,
        model: BayesianRewardModel,
        proxy_value: ProxyValueFunction,
    ) -> None:
        """Initialise the nightly updater.

        Args:
            model: BayesianRewardModel to update.
            proxy_value: ProxyValueFunction to update.
        """
        self.model = model
        self.proxy_value = proxy_value

        logger.debug("NightlyUpdater initialised")

    def update(self, day_data: list[Observation]) -> dict:
        """Process one day's data and update all model components.

        Args:
            day_data: List of dicts, each with keys:
                'g' (np.ndarray), 'f' (np.ndarray), 'action' (int),
                'pi' (float), 'reward' (float), 'available' (bool).

        Returns:
            Dict with keys:
                'n_observations': Total observations in the day.
                'n_available': Number of available decision times.
                'posterior_mean': Updated posterior mean.
                'posterior_cov': Updated posterior covariance.
        """
        n_available = 0

        for obs in day_data:
            self.model.update(
                g=obs["g"],
                f=obs["f"],
                action=obs["action"],
                pi=obs["pi"],
                reward=obs["reward"],
                available=obs["available"],
            )
            if obs["available"]:
                n_available += 1

        self.proxy_value.solve()
        self.proxy_value.update()

        result = {
            "n_observations": len(day_data),
            "n_available": n_available,
            "posterior_mean": self.model.posterior_mean.copy(),
            "posterior_cov": self.model.posterior_cov.copy(),
        }

        logger.debug(
            "Nightly update complete: %d obs, %d available",
            len(day_data),
            n_available,
        )

        return result
