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

    def update_batch(self, transitions: list[tuple]) -> None:
        """Batch update over daily transitions.

        Each transition: (phi, action, reward) where phi is the feature vector.
        For one-vs-rest: when reference action is taken, update ALL posteriors
        with negated features/reward. When non-reference action a is taken,
        update only posterior for a.

        Uses diagonal of outer product to avoid cross-correlations between
        independent one-hot feature groups (step_bin, sleep, etc.).
        """
        for phi, action, reward in transitions:
            phi_arr = np.asarray(phi, dtype=np.float64)
            diag = np.diag(phi_arr * phi_arr)
            if action == self._reference_action:
                for targeted in self._non_ref:
                    self._precision[targeted] += diag / self._sigma_sq
                    self._precision_mean[targeted] += -phi_arr * reward / self._sigma_sq
            else:
                self._precision[action] += diag / self._sigma_sq
                self._precision_mean[action] += phi_arr * reward / self._sigma_sq

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

    def get_reward_means(
        self, avg_features: np.ndarray | None = None
    ) -> dict[str, float]:
        """Expected reward for each action, marginalized over context.

        Uses avg_features (running mean of observed f(s)) if provided,
        otherwise returns 0 for all actions.
        """
        means = self.get_beta_means()
        f = avg_features if avg_features is not None else np.zeros(self._n_features)
        return {a: float(np.dot(means[a], f)) for a in self._actions}
