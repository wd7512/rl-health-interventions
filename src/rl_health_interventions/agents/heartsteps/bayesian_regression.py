from __future__ import annotations

import numpy as np


class MultiClassBayesianRegression:
    """Bayesian linear regression with action-centering for K actions.

    Maintains K-1 posteriors for beta_k where r(s, a_k) - r(s, a_0) = f(s)^T beta_k.
    Uses precision matrix representation for efficient incremental updates.
    """

    def __init__(
        self,
        n_features: int,
        actions: list[str],
        reference_action: str,
        sigma_sq: float = 1.0,
        prior_mean: float = 0.0,
        prior_cov: float = 1.0,
    ) -> None:
        if sigma_sq <= 0:
            msg = "sigma_sq must be > 0"
            raise ValueError(msg)
        if prior_cov <= 0:
            msg = "prior_cov must be > 0"
            raise ValueError(msg)
        self._n_features = n_features
        self._sigma_sq = sigma_sq
        self._actions = list(actions)
        self._reference_action = reference_action
        self._non_ref = [a for a in self._actions if a != reference_action]

        self._precision: dict[str, np.ndarray] = {}
        self._precision_mean: dict[str, np.ndarray] = {}

        prior_precision = np.eye(n_features) / prior_cov
        for action in self._non_ref:
            self._precision[action] = prior_precision.copy()
            self._precision_mean[action] = np.ones(n_features) * prior_mean / prior_cov

    @property
    def non_reference_actions(self) -> list[str]:
        return list(self._non_ref)

    def _design_vector(
        self, phi: np.ndarray, action: str, reward: float
    ) -> tuple[str, np.ndarray, float]:
        """Build the centered design vector for one-vs-rest."""
        if action == self._reference_action:
            return self._non_ref[0], -phi, -reward
        return action, phi, reward

    def update_batch(self, transitions: list[tuple]) -> None:
        """Batch update over daily transitions.

        Each transition: (phi, action, reward) where phi is the feature vector.
        """
        for phi, action, reward in transitions:
            phi_arr = np.asarray(phi, dtype=np.float64)
            targeted_action, centered_phi, centered_reward = self._design_vector(
                phi_arr, action, reward
            )
            outer = np.outer(centered_phi, centered_phi)
            self._precision[targeted_action] += outer / self._sigma_sq
            self._precision_mean[targeted_action] += (
                centered_phi * centered_reward / self._sigma_sq
            )

    def sample_betas(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Draw samples from each posterior."""
        samples: dict[str, np.ndarray] = {}
        for action in self._non_ref:
            cov = np.linalg.inv(self._precision[action])
            mean = cov @ self._precision_mean[action]
            samples[action] = rng.multivariate_normal(mean, cov)
        samples[self._reference_action] = np.zeros(self._n_features)
        return samples

    def get_beta_means(self) -> dict[str, np.ndarray]:
        """Posterior means for each action."""
        means: dict[str, np.ndarray] = {}
        for action in self._non_ref:
            cov = np.linalg.inv(self._precision[action])
            means[action] = cov @ self._precision_mean[action]
        means[self._reference_action] = np.zeros(self._n_features)
        return means

    def get_reward_means(self) -> dict[str, float]:
        """Expected reward for each action, averaged over feature distribution."""
        means = self.get_beta_means()
        return {
            a: float(np.dot(means[a], np.zeros(self._n_features)))
            for a in self._actions
        }
