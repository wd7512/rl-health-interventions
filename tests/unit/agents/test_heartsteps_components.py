"""Tests for HeartSteps agent components."""

from __future__ import annotations

import numpy as np
import pytest

from rl_health_interventions.agents.heartsteps.bayesian import BayesianRewardModel
from rl_health_interventions.agents.heartsteps.dosage import DosageTracker
from rl_health_interventions.agents.heartsteps.features import (
    construct_heartsteps_features,
)
from rl_health_interventions.agents.heartsteps.proxy import ProxyValueFunction


class TestDosageTracker:
    def test_initial_value_is_zero(self) -> None:
        d = DosageTracker(decay=0.95)
        assert d.value == 0.0

    def test_treatment_increases_dosage(self) -> None:
        d = DosageTracker(decay=0.95)
        d.update(treatment_delivered=True)
        assert d.value == 1.0

    def test_no_event_decays(self) -> None:
        d = DosageTracker(decay=0.95)
        d.update(treatment_delivered=True)
        d.update()
        assert abs(d.value - 0.95) < 1e-10

    def test_anti_sedentary_counts(self) -> None:
        d = DosageTracker(decay=0.95)
        d.update(anti_sedentary_delivered=True)
        assert d.value == 1.0

    def test_reset(self) -> None:
        d = DosageTracker(decay=0.95)
        d.update(treatment_delivered=True)
        d.reset()
        assert d.value == 0.0

    def test_invalid_decay_raises(self) -> None:
        with pytest.raises(ValueError):
            DosageTracker(decay=0.0)


class TestBayesianRewardModel:
    def _make_model(self, g_dim: int = 6, f_dim: int = 4) -> BayesianRewardModel:
        total = g_dim + 2 * f_dim
        return BayesianRewardModel(
            g_dim=g_dim,
            f_dim=f_dim,
            prior_mean=np.zeros(total),
            prior_cov=np.eye(total),
            noise_variance=1.0,
        )

    def test_shape_validation(self) -> None:
        with pytest.raises(ValueError):
            BayesianRewardModel(6, 4, np.zeros(10), np.eye(10), 1.0)

    def test_update_posterior(self) -> None:
        m = self._make_model()
        old_mean = m.posterior_mean.copy()
        g = np.random.default_rng(0).normal(size=6)
        f = np.random.default_rng(1).normal(size=4)
        m.update(g, f, action=1, pi=0.3, reward=1.0, available=True)
        assert not np.array_equal(m.posterior_mean, old_mean)

    def test_update_skipped_when_unavailable(self) -> None:
        m = self._make_model()
        old_mean = m.posterior_mean.copy()
        g = np.zeros(6)
        f = np.zeros(4)
        m.update(g, f, action=1, pi=0.3, reward=1.0, available=False)
        assert np.array_equal(m.posterior_mean, old_mean)

    def test_sample_beta_shape(self) -> None:
        m = self._make_model()
        beta = m.sample_beta(np.random.default_rng(42))
        assert beta.shape == (4,)


class TestProxyValueFunction:
    def test_solve_converges(self) -> None:
        pv = ProxyValueFunction(gamma=0.9, grid_max=5.0, grid_step=0.5)
        result = pv.solve()
        assert result["converged"]

    def test_H_returns_float(self) -> None:
        pv = ProxyValueFunction(gamma=0.9, grid_max=5.0, grid_step=0.5)
        pv.solve()
        h = pv.H(1.0, 0)
        assert isinstance(h, float)

    def test_high_dosage_reduces_action_value(self) -> None:
        pv = ProxyValueFunction(gamma=0.9, grid_max=20.0, grid_step=0.5)
        pv.solve()
        h_low = pv.H(0.0, 1)
        h_high = pv.H(15.0, 1)
        assert h_low >= h_high


class TestConstructFeatures:
    def test_returns_correct_shapes(self) -> None:
        steps = np.arange(100, dtype=float)
        g, f = construct_heartsteps_features(steps, 50, 5000.0, 2, 5)
        assert g.shape == (6,)
        assert f.shape == (4,)

    def test_normalisation_bounds(self) -> None:
        steps = np.ones(100) * 10000
        g, f = construct_heartsteps_features(steps, 50, 50000.0, 2, 5)
        # steps_30min_norm and daily_norm are clamped to [0, 1]
        assert 0.0 <= g[0] <= 1.0
        assert 0.0 <= g[1] <= 1.0
        # sin/cos are in [-1, 1]
        assert -1.0 <= g[2] <= 1.0
        assert -1.0 <= g[3] <= 1.0
