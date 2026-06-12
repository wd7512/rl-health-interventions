"""Thompson Sampling Bandit baseline without proxy value.

Uses TS to maximise immediate reward only (no proxy for delayed effects).
No action-centering in the reward model.

Action selection:
    1. Sample theta from posterior N(mu, Sigma).
    2. Compute Q(s, a) = expected immediate reward for each action.
    3. Softmax with temperature, clip to [epsilon_1, epsilon_0].
    4. Sample A ~ Bernoulli(pi).

Reference:
    Liao et al. (2019). arXiv:1909.03539, Section 6.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class TSBandit:
    """Thompson Sampling bandit maximising immediate reward.

    Reward model (no action-centering):
        R_t = g(S_t)^T alpha + A_t * f(S_t)^T beta + N(0, sigma^2)

    Attributes:
        g_dim: Baseline feature dimension.
        f_dim: Treatment feature dimension.
        total_params: g_dim + f_dim.
        noise_variance: Fixed observation noise variance.
    """

    def __init__(
        self,
        g_dim: int,
        f_dim: int,
        prior_mean: np.ndarray,
        prior_cov: np.ndarray,
        noise_variance: float,
        tau: float = 1.0,
        epsilon_0: float = 0.2,
        epsilon_1: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.g_dim = g_dim
        self.f_dim = f_dim
        self.total_params = g_dim + f_dim
        self.noise_variance = noise_variance
        self.tau = tau
        self.epsilon_0 = epsilon_0
        self.epsilon_1 = epsilon_1
        self._rng = rng if rng is not None else np.random.default_rng()

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

        logger.debug(
            "TSBandit initialised: g_dim=%d, f_dim=%d, tau=%.2f",
            g_dim,
            f_dim,
            tau,
        )

    def _construct_features(self, g: np.ndarray, f: np.ndarray, action: int) -> np.ndarray:
        return np.concatenate([g, action * f])

    def select_action(
        self,
        g: np.ndarray,
        f: np.ndarray,
        available: bool,
    ) -> tuple[int, float]:
        if not available:
            return 0, 0.0

        theta_sample = self._rng.multivariate_normal(
            self._posterior_mean, self._posterior_cov,
        )

        alpha_sample = theta_sample[: self.g_dim]
        beta_sample = theta_sample[self.g_dim :]

        Q0 = float(g @ alpha_sample)
        Q1 = float(g @ alpha_sample + f @ beta_sample)

        exp0 = np.exp(Q0 / self.tau)
        exp1 = np.exp(Q1 / self.tau)
        prob_raw = exp1 / (exp0 + exp1)

        prob = float(np.clip(prob_raw, self.epsilon_1, self.epsilon_0))
        action = int(self._rng.binomial(1, prob))

        return action, prob

    def step(
        self,
        g: np.ndarray,
        f: np.ndarray,
        action: int,
        pi: float,
        reward: float,
        available: bool = True,
    ) -> None:
        if not available:
            logger.debug("TSBandit skipping update: unavailable")
            return

        phi = self._construct_features(g, f, action)
        sigma2_inv = 1.0 / self.noise_variance

        outer = np.outer(phi, phi)
        Sigma_prior_inv = np.linalg.inv(self._posterior_cov)
        new_cov = np.linalg.inv(Sigma_prior_inv + sigma2_inv * outer)

        new_mean = new_cov @ (
            sigma2_inv * phi * reward + Sigma_prior_inv @ self._posterior_mean
        )

        self._posterior_mean = new_mean
        self._posterior_cov = new_cov

        logger.debug(
            "TSBandit posterior updated: ||mean||=%.4f",
            np.linalg.norm(self._posterior_mean),
        )
