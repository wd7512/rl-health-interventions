"""Tests for nightly batch update (Paper Section 5.4)."""

import numpy as np
import pytest
from paper_reproduction.heartsteps.bayesian_regression import BayesianRewardModel
from paper_reproduction.heartsteps.nightly_update import NightlyUpdater
from paper_reproduction.heartsteps.proxy_value import ProxyValueFunction


class TestModelUpdate:
    def test_correct_posterior(self):
        g_dim, f_dim = 2, 1
        total = g_dim + 2 * f_dim
        model = BayesianRewardModel(
            g_dim=g_dim, f_dim=f_dim,
            prior_mean=np.zeros(total),
            prior_cov=np.eye(total),
            noise_variance=1.0,
        )
        proxy = ProxyValueFunction()
        proxy.solve()
        updater = NightlyUpdater(model=model, proxy_value=proxy)

        daily_data = [
            {"g": np.array([1.0, 0.0]), "f": np.array([1.0]),
             "action": 1, "pi": 0.3, "reward": 5.0, "available": True},
            {"g": np.array([0.0, 1.0]), "f": np.array([1.0]),
             "action": 0, "pi": 0.3, "reward": 1.0, "available": True},
            {"g": np.array([0.5, 0.5]), "f": np.array([1.0]),
             "action": 0, "pi": 0.3, "reward": 2.0, "available": False},
            {"g": np.array([1.0, 1.0]), "f": np.array([1.0]),
             "action": 1, "pi": 0.3, "reward": 3.0, "available": True},
            {"g": np.array([0.0, 0.0]), "f": np.array([1.0]),
             "action": 1, "pi": 0.3, "reward": 4.0, "available": False},
        ]

        updater.update(daily_data)

        model_manual = BayesianRewardModel(
            g_dim=g_dim, f_dim=f_dim,
            prior_mean=np.zeros(total),
            prior_cov=np.eye(total),
            noise_variance=1.0,
        )
        for obs in [daily_data[0], daily_data[1], daily_data[3]]:
            model_manual.update(
                obs["g"], obs["f"], obs["action"], obs["pi"],
                obs["reward"], available=True,
            )

        np.testing.assert_allclose(
            model.posterior_mean, model_manual.posterior_mean, atol=1e-10,
        )
        np.testing.assert_allclose(
            model.posterior_cov, model_manual.posterior_cov, atol=1e-10,
        )

    def test_variance_decreases_over_days(self):
        g_dim, f_dim = 2, 1
        total = g_dim + 2 * f_dim
        model = BayesianRewardModel(
            g_dim=g_dim, f_dim=f_dim,
            prior_mean=np.zeros(total),
            prior_cov=np.eye(total),
            noise_variance=1.0,
        )
        proxy = ProxyValueFunction()
        proxy.solve()
        updater = NightlyUpdater(model=model, proxy_value=proxy)
        trace_before = np.trace(model.posterior_cov)
        rng = np.random.default_rng(42)
        for _ in range(5):
            daily_data = []
            for _ in range(5):
                daily_data.append({
                    "g": np.array([1.0, 0.0]), "f": np.array([1.0]),
                    "action": rng.binomial(1, 0.3), "pi": 0.3,
                    "reward": rng.normal(5.0),
                    "available": bool(rng.binomial(1, 0.8)),
                })
            updater.update(daily_data)
        trace_after = np.trace(model.posterior_cov)
        assert trace_after < trace_before

    def test_unavailable_times_skipped(self):
        g_dim, f_dim = 2, 1
        total = g_dim + 2 * f_dim
        model = BayesianRewardModel(
            g_dim=g_dim, f_dim=f_dim,
            prior_mean=np.zeros(total),
            prior_cov=np.eye(total),
            noise_variance=1.0,
        )
        proxy = ProxyValueFunction()
        proxy.solve()
        updater = NightlyUpdater(model=model, proxy_value=proxy)
        mean_before = model.posterior_mean.copy()
        daily_data = [
            {"g": np.array([1.0, 0.0]), "f": np.array([1.0]),
             "action": 1, "pi": 0.3, "reward": 5.0, "available": False}
            for _ in range(5)
        ]
        updater.update(daily_data)
        np.testing.assert_allclose(model.posterior_mean, mean_before)

    def test_proxy_value_update_called(self):
        proxy = ProxyValueFunction()
        proxy.solve()
        H_before = proxy._H_current.copy()
        g_dim, f_dim = 2, 1
        total = g_dim + 2 * f_dim
        model = BayesianRewardModel(
            g_dim=g_dim, f_dim=f_dim,
            prior_mean=np.zeros(total),
            prior_cov=np.eye(total) * 10,
            noise_variance=1.0,
        )
        updater = NightlyUpdater(model=model, proxy_value=proxy)
        rng = np.random.default_rng(42)
        daily_data = []
        for _ in range(5):
            daily_data.append({
                "g": np.array([1.0, 0.0]), "f": np.array([1.0]),
                "action": rng.binomial(1, 0.3), "pi": 0.3,
                "reward": rng.normal(5.0), "available": True,
            })
        updater.update(daily_data)
        assert not np.allclose(proxy._H_current, H_before)

    def test_returns_summary_dict(self):
        g_dim, f_dim = 2, 1
        total = g_dim + 2 * f_dim
        model = BayesianRewardModel(
            g_dim=g_dim, f_dim=f_dim,
            prior_mean=np.zeros(total),
            prior_cov=np.eye(total),
            noise_variance=1.0,
        )
        proxy = ProxyValueFunction()
        proxy.solve()
        updater = NightlyUpdater(model=model, proxy_value=proxy)
        daily_data = [
            {"g": np.array([1.0, 0.0]), "f": np.array([1.0]),
             "action": 1, "pi": 0.3, "reward": 5.0, "available": True}
            for _ in range(5)
        ]
        result = updater.update(daily_data)
        assert isinstance(result, dict)
        assert result["n_observations"] == 5
