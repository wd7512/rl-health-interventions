"""Bayesian linear regression with action-centering for treatment effect estimation.

Implements the working model from Section 5.4.1 of the HeartSteps V2 paper:

    R_t = g(S_t)^T alpha_0 + pi_t * f(S_t)^T alpha_1 + (A_t - pi_t) * f(S_t)^T beta + N(0, sigma^2)

Key insight: Action-centering via (A_t - pi_t) makes the treatment effect
estimate beta robust to misspecification of the baseline reward model g(s).
Even if g(s) is wrong, beta converges to the true treatment effect.

The joint parameter vector is theta = [alpha_0, alpha_1, beta] and the joint
feature vector is:
    phi(S, A) = [g(S)^T, pi * f(S)^T, (A - pi) * f(S)^T]

Posterior update (equations 5-6):
    Sigma_{d+1} = (sum phi*phi^T / sigma^2 + Sigma_prior^{-1})^{-1}
    mu_{d+1} = Sigma_{d+1} * (sum phi*R / sigma^2 + Sigma_prior^{-1} * mu_prior)

Reference:
    Liao et al. (2019). arXiv:1909.03539, Section 5.4.1.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class BayesianRewardModel:
    """Bayesian linear regression with action-centering.

    Tracks the posterior distribution over parameters theta = [alpha_0, alpha_1, beta]
    where:
        - alpha_0: baseline reward parameters (g_dim)
        - alpha_1: main effect parameters scaled by pi_t (f_dim)
        - beta: treatment effect parameters (f_dim)

    The treatment effect beta is extracted as the last f_dim elements of the
    posterior mean.

    Attributes:
        g_dim: Dimensionality of baseline feature vector g(s).
        f_dim: Dimensionality of treatment effect feature vector f(s).
        total_params: Total parameter count (g_dim + 2 * f_dim).
        noise_variance: Fixed noise variance sigma^2 (not learned).
        posterior_mean: Current posterior mean mu.
        posterior_cov: Current posterior covariance Sigma.
    """

    def __init__(
        self,
        g_dim: int,
        f_dim: int,
        prior_mean: np.ndarray,
        prior_cov: np.ndarray,
        noise_variance: float,
    ) -> None:
        """Initialise with prior distribution.

        Args:
            g_dim: Dimension of baseline features g(s).
            f_dim: Dimension of treatment effect features f(s).
            prior_mean: Prior mean vector of length g_dim + 2*f_dim.
            prior_cov: Prior covariance matrix of shape (g_dim+2*f_dim, g_dim+2*f_dim).
            noise_variance: Fixed noise variance sigma^2.
        """
        self.g_dim = g_dim
        self.f_dim = f_dim
        self.total_params = g_dim + 2 * f_dim
        self.noise_variance = noise_variance

        if prior_mean.shape != (self.total_params,):
            msg = f"prior_mean shape {prior_mean.shape} != ({self.total_params},)"
            raise ValueError(msg)
        if prior_cov.shape != (self.total_params, self.total_params):
            msg = f"prior_cov shape {prior_cov.shape} != ({self.total_params}, {self.total_params})"
            raise ValueError(msg)

        self._posterior_mean = prior_mean.copy()
        self._posterior_cov = prior_cov.copy()

        logger.debug(
            "BayesianRewardModel initialised: g_dim=%d, f_dim=%d, sigma2=%.4f",
            g_dim,
            f_dim,
            noise_variance,
        )

    @property
    def posterior_mean(self) -> np.ndarray:
        """Current posterior mean mu."""
        return self._posterior_mean

    @property
    def posterior_cov(self) -> np.ndarray:
        """Current posterior covariance Sigma."""
        return self._posterior_cov

    @property
    def beta_mean(self) -> np.ndarray:
        """Treatment effect parameters (last f_dim elements of posterior mean)."""
        return self._posterior_mean[-self.f_dim :]

    @property
    def beta_cov(self) -> np.ndarray:
        """Covariance of treatment effect parameters."""
        return self._posterior_cov[-self.f_dim :, -self.f_dim :]

    def construct_features(
        self,
        g: np.ndarray,
        f: np.ndarray,
        action: int,
        pi: float,
    ) -> np.ndarray:
        """Construct the joint feature vector phi(S, A).

        phi(S, A) = [g(S)^T, pi * f(S)^T, (A - pi) * f(S)^T]

        Args:
            g: Baseline feature vector of length g_dim.
            f: Treatment effect feature vector of length f_dim.
            action: Binary action A_t in {0, 1}.
            pi: Current randomization probability.

        Returns:
            Joint feature vector of length g_dim + 2*f_dim.
        """
        centering = action - pi
        phi = np.concatenate([g, pi * f, centering * f])
        return phi

    def update(
        self,
        g: np.ndarray,
        f: np.ndarray,
        action: int,
        pi: float,
        reward: float,
        available: bool = True,
    ) -> None:
        """Bayesian posterior update with one observation.

        Only updates when the participant is available (I_t = 1).
        Uses equations (5)-(6) from the paper.

        Args:
            g: Baseline features for this observation.
            f: Treatment features for this observation.
            action: Binary action A_t.
            pi: Randomization probability at this time.
            reward: Observed reward R_t.
            available: Whether participant was available (I_t = 1).
        """
        if not available:
            logger.debug("Skipping update: participant unavailable")
            return

        phi = self.construct_features(g, f, action, pi)
        sigma2_inv = 1.0 / self.noise_variance

        # Equation (5): Sigma_{d+1} = (phi * phi^T / sigma^2 + Sigma_prior^{-1})^{-1}
        outer = np.outer(phi, phi)
        Sigma_prior_inv = np.linalg.inv(self._posterior_cov)
        new_cov = np.linalg.inv(Sigma_prior_inv + sigma2_inv * outer)

        # Equation (6): mu_{d+1} = Sigma_{d+1} * (phi * R / sigma^2 + Sigma_prior^{-1} * mu_prior)
        new_mean = new_cov @ (
            sigma2_inv * phi * reward + Sigma_prior_inv @ self._posterior_mean
        )

        self._posterior_mean = new_mean
        self._posterior_cov = new_cov

        logger.debug(
            "Posterior updated: ||beta||=%.4f, trace(Sigma)=%.4f",
            np.linalg.norm(self.beta_mean),
            np.trace(self._posterior_cov),
        )

    def expected_reward(
        self,
        g: np.ndarray,
        f: np.ndarray,
        action: int,
        pi: float,
    ) -> float:
        """Compute expected reward E[R | S, A] under current posterior mean.

        Uses the action-centered model:
            E[R] = g^T * alpha_0_hat + pi * f^T * alpha_1_hat + (A - pi) * f^T * beta_hat

        Args:
            g: Baseline features.
            f: Treatment features.
            action: Binary action.
            pi: Randomization probability.

        Returns:
            Expected reward under current parameter estimates.
        """
        phi = self.construct_features(g, f, action, pi)
        return float(phi @ self._posterior_mean)

    def sample_beta(self, rng: np.random.Generator) -> np.ndarray:
        """Sample treatment effect parameters from the posterior.

        Used for Thompson Sampling action selection.

        Args:
            rng: NumPy random generator.

        Returns:
            Sampled beta vector of length f_dim.
        """
        return rng.multivariate_normal(self.beta_mean, self.beta_cov)
