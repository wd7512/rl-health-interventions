"""Tests for tuning parameter grid search (Task 3.3)."""

import numpy as np
import pytest

from paper_reproduction.simulation.tuning import grid_search


@pytest.fixture
def step_data():
    rng = np.random.default_rng(42)
    return rng.poisson(lam=50, size=(6, 7, 10)).astype(float)


@pytest.fixture
def alpha():
    return np.array([1.0, 0.5, 0.3, 0.1, 0.2, 0.1, 0.2, 0.15, 0.1, 0.05])


@pytest.fixture
def beta():
    return np.array([0.5, 0.3, 0.2, 0.1])


@pytest.fixture
def prior_mean():
    return np.zeros(14)


@pytest.fixture
def prior_cov():
    return np.eye(14)


class TestGridSearchCompletes:
    def test_returns_dict(self, step_data, alpha, beta, prior_mean, prior_cov):
        train_idx = np.array([0, 1, 2])
        result = grid_search(
            step_data=step_data,
            train_indices=train_idx,
            alpha=alpha,
            beta=beta,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            n_re_runs=2,
            g_dim=6,
            f_dim=4,
            seed=42,
        )
        assert isinstance(result, dict)

    def test_expected_keys(self, step_data, alpha, beta, prior_mean, prior_cov):
        train_idx = np.array([0, 1, 2])
        result = grid_search(
            step_data=step_data,
            train_indices=train_idx,
            alpha=alpha,
            beta=beta,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            n_re_runs=2,
            g_dim=6,
            f_dim=4,
            seed=42,
        )
        expected_keys = {"best_gamma", "best_w", "best_reward", "grid_results"}
        assert expected_keys.issubset(result.keys())

    def test_full_grid_evaluated(self, step_data, alpha, beta, prior_mean, prior_cov):
        train_idx = np.array([0])
        result = grid_search(
            step_data=step_data,
            train_indices=train_idx,
            alpha=alpha,
            beta=beta,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            n_re_runs=2,
            g_dim=6,
            f_dim=4,
            seed=42,
        )
        assert len(result["grid_results"]) == 36


class TestSelectedParameters:
    def test_gamma_from_grid(self, step_data, alpha, beta, prior_mean, prior_cov):
        train_idx = np.array([0, 1])
        result = grid_search(
            step_data=step_data,
            train_indices=train_idx,
            alpha=alpha,
            beta=beta,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            n_re_runs=2,
            g_dim=6,
            f_dim=4,
            seed=42,
        )
        assert result["best_gamma"] in {0, 0.25, 0.5, 0.75, 0.9, 0.95}

    def test_w_from_grid(self, step_data, alpha, beta, prior_mean, prior_cov):
        train_idx = np.array([0, 1])
        result = grid_search(
            step_data=step_data,
            train_indices=train_idx,
            alpha=alpha,
            beta=beta,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            n_re_runs=2,
            g_dim=6,
            f_dim=4,
            seed=42,
        )
        assert result["best_w"] in {0, 0.1, 0.25, 0.5, 0.75, 1.0}

    def test_best_reward_matches_grid(
        self, step_data, alpha, beta, prior_mean, prior_cov
    ):
        train_idx = np.array([0, 1])
        result = grid_search(
            step_data=step_data,
            train_indices=train_idx,
            alpha=alpha,
            beta=beta,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            n_re_runs=2,
            g_dim=6,
            f_dim=4,
            seed=42,
        )
        key = (result["best_gamma"], result["best_w"])
        assert key in result["grid_results"]
        assert result["grid_results"][key] == pytest.approx(result["best_reward"])


class TestDeterminism:
    def test_same_seed_identical(self, step_data, alpha, beta, prior_mean, prior_cov):
        train_idx = np.array([0, 1, 2])
        r1 = grid_search(
            step_data=step_data,
            train_indices=train_idx,
            alpha=alpha,
            beta=beta,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            n_re_runs=2,
            g_dim=6,
            f_dim=4,
            seed=42,
        )
        r2 = grid_search(
            step_data=step_data,
            train_indices=train_idx,
            alpha=alpha,
            beta=beta,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            n_re_runs=2,
            g_dim=6,
            f_dim=4,
            seed=42,
        )
        assert r1["best_gamma"] == r2["best_gamma"]
        assert r1["best_w"] == r2["best_w"]
        assert r1["best_reward"] == pytest.approx(r2["best_reward"])

    def test_different_seeds_differ(
        self, step_data, alpha, beta, prior_mean, prior_cov
    ):
        train_idx = np.array([0, 1])
        r1 = grid_search(
            step_data=step_data,
            train_indices=train_idx,
            alpha=alpha,
            beta=beta,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            n_re_runs=2,
            g_dim=6,
            f_dim=4,
            seed=42,
        )
        r2 = grid_search(
            step_data=step_data,
            train_indices=train_idx,
            alpha=alpha,
            beta=beta,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            n_re_runs=2,
            g_dim=6,
            f_dim=4,
            seed=99,
        )
        different = (
            r1["best_gamma"] != r2["best_gamma"]
            or abs(r1["best_reward"] - r2["best_reward"]) > 1e-6
        )
        assert different
