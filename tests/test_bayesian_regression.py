"""Tests for Bayesian linear regression with action-centering (Paper Section 5.4.1).

The working model:
    R_t = g(S_t)^T alpha_0 + pi_t * f(S_t)^T alpha_1 + (A_t - pi_t) * f(S_t)^T beta + N(0, sigma^2)

Action-centering via (A_t - pi_t) makes treatment effect estimates robust to
baseline model misspecification — even if g(s) is wrong, beta is unbiased.

Reference:
    Liao et al. (2019). arXiv:1909.03539, Section 5.4.1, equations (3)-(6).
"""

import numpy as np
import pytest

from paper_reproduction.heartsteps.bayesian_regression import BayesianRewardModel


class TestModelInitialization:
    def test_creates_with_prior(self):
        model = BayesianRewardModel(
            g_dim=3,
            f_dim=2,
            prior_mean=np.zeros(7),  # g_dim + 2*f_dim = 3 + 4 = 7
            prior_cov=np.eye(7),
            noise_variance=1.0,
        )
        assert model.g_dim == 3
        assert model.f_dim == 2
        assert model.total_params == 7

    def test_prior_shapes(self):
        g_dim, f_dim = 4, 3
        model = BayesianRewardModel(
            g_dim=g_dim,
            f_dim=f_dim,
            prior_mean=np.zeros(g_dim + 2 * f_dim),
            prior_cov=np.eye(g_dim + 2 * f_dim),
            noise_variance=1.0,
        )
        assert model.posterior_mean.shape == (g_dim + 2 * f_dim,)
        assert model.posterior_cov.shape == (g_dim + 2 * f_dim, g_dim + 2 * f_dim)


class TestFeatureConstruction:
    def test_joint_feature_vector(self):
        """phi(S, A) = [g(S)^T, pi * f(S)^T, (A - pi) * f(S)^T]"""
        model = BayesianRewardModel(
            g_dim=2,
            f_dim=1,
            prior_mean=np.zeros(4),
            prior_cov=np.eye(4),
            noise_variance=1.0,
        )
        g = np.array([1.0, 0.5])  # baseline features
        f = np.array([0.8])  # treatment features
        A = 1
        pi = 0.3

        phi = model.construct_features(g, f, A, pi)

        expected = np.array([1.0, 0.5, 0.3 * 0.8, (1.0 - 0.3) * 0.8])
        np.testing.assert_allclose(phi, expected)

    def test_action_centering_at_pi(self):
        """When A = pi, the (A-pi)*f term vanishes but pi*f remains."""
        model = BayesianRewardModel(
            g_dim=2,
            f_dim=2,
            prior_mean=np.zeros(6),
            prior_cov=np.eye(6),
            noise_variance=1.0,
        )
        g = np.array([1.0, 0.5])
        f = np.array([0.8, 0.3])
        A = 0.5
        pi = 0.5

        phi = model.construct_features(g, f, A, pi)
        # phi = [g, pi*f, (A-pi)*f] = [1.0, 0.5, 0.4, 0.15, 0.0, 0.0]
        assert phi[0] == pytest.approx(1.0)  # g[0]
        assert phi[1] == pytest.approx(0.5)  # g[1]
        assert phi[2] == pytest.approx(0.4)  # pi * f[0] = 0.5 * 0.8
        assert phi[3] == pytest.approx(0.15)  # pi * f[1] = 0.5 * 0.3
        assert phi[4] == pytest.approx(0.0)  # (A-pi) * f[0] = 0
        assert phi[5] == pytest.approx(0.0)  # (A-pi) * f[1] = 0


class TestPosteriorUpdate:
    def test_single_observation_update(self):
        """Posterior update with one observation matches hand calculation."""
        g_dim, f_dim = 2, 1
        total = g_dim + 2 * f_dim
        sigma2 = 1.0

        model = BayesianRewardModel(
            g_dim=g_dim,
            f_dim=f_dim,
            prior_mean=np.zeros(total),
            prior_cov=np.eye(total),
            noise_variance=sigma2,
        )

        # One observation: g=[1,0], f=[1], A=1, pi=0.3, R=5.0
        g = np.array([1.0, 0.0])
        f = np.array([1.0])
        A, pi, R = 1, 0.3, 5.0
        phi = model.construct_features(g, f, A, pi)

        # Manual computation of posterior
        Sigma_prior_inv = np.eye(total)
        outer = np.outer(phi, phi)
        Sigma_post = np.linalg.inv(Sigma_prior_inv + (1 / sigma2) * outer)
        mu_post = Sigma_post @ (
            (1 / sigma2) * phi * R + Sigma_prior_inv @ np.zeros(total)
        )

        # Update model
        model.update(g, f, A, pi, R, available=True)

        np.testing.assert_allclose(model.posterior_mean, mu_post, atol=1e-10)
        np.testing.assert_allclose(model.posterior_cov, Sigma_post, atol=1e-10)

    def test_unavailable_observation_does_not_update(self):
        model = BayesianRewardModel(
            g_dim=2,
            f_dim=1,
            prior_mean=np.zeros(4),
            prior_cov=np.eye(4),
            noise_variance=1.0,
        )
        mean_before = model.posterior_mean.copy()
        model.update(
            g=np.array([1.0, 0.0]),
            f=np.array([1.0]),
            action=0,
            pi=0.3,
            reward=5.0,
            available=False,
        )
        np.testing.assert_allclose(model.posterior_mean, mean_before)

    def test_posterior_variance_decreases_with_data(self):
        model = BayesianRewardModel(
            g_dim=2,
            f_dim=1,
            prior_mean=np.zeros(4),
            prior_cov=np.eye(4),
            noise_variance=1.0,
        )
        trace_before = np.trace(model.posterior_cov)

        for _ in range(10):
            model.update(
                g=np.array([1.0, 0.0]),
                f=np.array([1.0]),
                action=np.random.binomial(1, 0.3),
                pi=0.3,
                reward=np.random.normal(5.0),
                available=True,
            )

        trace_after = np.trace(model.posterior_cov)
        assert trace_after < trace_before


class TestTreatmentEffectEstimation:
    def test_beta_extraction(self):
        """beta is the last f_dim elements of the posterior mean."""
        g_dim, f_dim = 2, 2
        model = BayesianRewardModel(
            g_dim=g_dim,
            f_dim=f_dim,
            prior_mean=np.zeros(g_dim + 2 * f_dim),
            prior_cov=np.eye(g_dim + 2 * f_dim),
            noise_variance=1.0,
        )
        assert model.beta_mean.shape == (f_dim,)
        assert model.beta_cov.shape == (f_dim, f_dim)

    def test_treatment_effect_robust_to_baseline_misspecification(self):
        """Action-centering: even with wrong g(s), beta estimate converges to truth."""
        rng = np.random.default_rng(42)
        g_dim, f_dim = 3, 2
        total = g_dim + 2 * f_dim
        sigma2 = 0.5

        # True parameters
        true_alpha = np.array([1.0, 2.0, -0.5])
        true_beta = np.array([3.0, -1.0])

        # Model with WRONG baseline features (only 2 of 3 true features)
        # This tests that action-centering still recovers beta
        model = BayesianRewardModel(
            g_dim=g_dim,
            f_dim=f_dim,
            prior_mean=np.zeros(total),
            prior_cov=np.eye(total) * 10,  # diffuse prior
            noise_variance=sigma2,
        )

        for _ in range(200):
            g = rng.normal(size=g_dim)
            f = rng.normal(size=f_dim)
            A = rng.binomial(1, 0.3)
            pi = 0.3
            # True reward uses all 3 baseline features
            R = g @ true_alpha + A * f @ true_beta + rng.normal(0, np.sqrt(sigma2))

            model.update(g, f, A, pi, R, available=True)

        # beta should be close to true values despite baseline misspecification
        beta_hat = model.beta_mean
        # With 200 observations and diffuse prior, should be within 1.0 of truth
        assert np.allclose(beta_hat, true_beta, atol=1.0), (
            f"beta_hat={beta_hat}, true_beta={true_beta}"
        )


class TestPrediction:
    def test_expected_reward_available(self):
        model = BayesianRewardModel(
            g_dim=2,
            f_dim=1,
            prior_mean=np.zeros(4),
            prior_cov=np.eye(4),
            noise_variance=1.0,
        )
        g = np.array([1.0, 0.5])
        f = np.array([0.8])

        # With zero prior, expected reward is 0 for any action
        r0 = model.expected_reward(g, f, action=0, pi=0.3)
        r1 = model.expected_reward(g, f, action=1, pi=0.3)
        assert r0 == pytest.approx(0.0)
        assert r1 == pytest.approx(0.0)
