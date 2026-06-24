"""Bayesian linear regression with action-centering for treatment effect estimation.

R_t = g(S)^T alpha_0 + pi * f(S)^T alpha_1 + (A - pi) * f(S)^T beta + eps
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class BayesianRewardModel:
    """Bayesian linear regression with action-centering.

    Joint parameter vector: theta = [alpha_0, alpha_1, beta].
    Joint feature vector: phi(S, A) = [g(S), pi*f(S), (A-pi)*f(S)].
    """

    def __init__(
        self,
        g_dim: int,
        f_dim: int,
        prior_mean: np.ndarray,
        prior_cov: np.ndarray,
        noise_variance: float,
    ) -> None:
        self.g_dim = g_dim
        self.f_dim = f_dim
        self.total_params = g_dim + 2 * f_dim
        self.noise_variance = noise_variance

        if prior_mean.shape != (self.total_params,):
            msg = f"prior_mean shape {prior_mean.shape} != ({self.total_params},)"
            raise ValueError(msg)
        if prior_cov.shape != (self.total_params, self.total_params):
            msg = (
                f"prior_cov shape {prior_cov.shape}"
                f" != ({self.total_params}, {self.total_params})"
            )
            raise ValueError(msg)

        self._posterior_mean = prior_mean.copy()
        self._posterior_cov = prior_cov.copy()

    @property
    def posterior_mean(self) -> np.ndarray:
        return self._posterior_mean

    @property
    def posterior_cov(self) -> np.ndarray:
        return self._posterior_cov

    @property
    def beta_mean(self) -> np.ndarray:
        return self._posterior_mean[-self.f_dim :]

    @property
    def beta_cov(self) -> np.ndarray:
        return self._posterior_cov[-self.f_dim :, -self.f_dim :]

    def construct_features(
        self, g: np.ndarray, f: np.ndarray, action: int, pi: float
    ) -> np.ndarray:
        centering = action - pi
        return np.concatenate([g, pi * f, centering * f])

    def update(
        self,
        g: np.ndarray,
        f: np.ndarray,
        action: int,
        pi: float,
        reward: float,
        available: bool = True,
    ) -> None:
        if not available:
            return

        phi = self.construct_features(g, f, action, pi)
        sigma2_inv = 1.0 / self.noise_variance
        outer = np.outer(phi, phi)
        Sigma_prior_inv = np.linalg.inv(self._posterior_cov)
        new_cov = np.linalg.inv(Sigma_prior_inv + sigma2_inv * outer)
        new_mean = new_cov @ (
            sigma2_inv * phi * reward + Sigma_prior_inv @ self._posterior_mean
        )
        self._posterior_mean = new_mean
        self._posterior_cov = new_cov

    def sample_beta(self, rng: np.random.Generator) -> np.ndarray:
        return rng.multivariate_normal(self.beta_mean, self.beta_cov)

    def copy(self) -> BayesianRewardModel:
        return BayesianRewardModel(
            g_dim=self.g_dim,
            f_dim=self.f_dim,
            prior_mean=self._posterior_mean.copy(),
            prior_cov=self._posterior_cov.copy(),
            noise_variance=self.noise_variance,
        )
