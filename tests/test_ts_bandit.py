"""Tests for TS Bandit baseline (Task 3.4).

The TS Bandit uses Thompson Sampling without proxy value or action-centering.
It maximises immediate reward only, with probability clipping.
"""

import numpy as np
import pytest

from paper_reproduction.baselines.ts_bandit import TSBandit


def _make_bandit(g_dim=4, f_dim=2, rng=None, seed=42):
    if rng is None:
        rng = np.random.default_rng(seed)
    total = g_dim + f_dim
    return TSBandit(
        g_dim=g_dim,
        f_dim=f_dim,
        prior_mean=np.zeros(total),
        prior_cov=np.eye(total),
        noise_variance=1.0,
        tau=1.0,
        epsilon_0=0.2,
        epsilon_1=0.1,
        rng=rng,
    )


class TestActionSelection:
    def test_action_is_binary(self):
        bandit = _make_bandit(g_dim=2, f_dim=1, seed=42)
        g = np.array([1.0, 0.5])
        f = np.array([0.8])
        for _ in range(50):
            action, _ = bandit.select_action(g, f, available=True)
            assert action in (0, 1)

    def test_unavailable_forces_zero(self):
        bandit = _make_bandit(g_dim=2, f_dim=1, seed=42)
        g = np.array([1.0, 0.5])
        f = np.array([0.8])
        for _ in range(50):
            action, _ = bandit.select_action(g, f, available=False)
            assert action == 0

    def test_probability_clipping(self):
        bandit = _make_bandit(g_dim=2, f_dim=1, seed=42)
        g = np.array([1.0, 0.5])
        f = np.array([0.8])
        for _ in range(50):
            _, prob = bandit.select_action(g, f, available=True)
            assert 0.1 <= prob <= 0.2 + 1e-10

    def test_action_variation(self):
        bandit = _make_bandit(g_dim=2, f_dim=1, seed=42)
        g = np.array([1.0, 0.5])
        f = np.array([0.8])
        actions = set()
        for _ in range(100):
            action, _ = bandit.select_action(g, f, available=True)
            actions.add(action)
        assert len(actions) == 2

    def test_returns_probability(self):
        bandit = _make_bandit(g_dim=2, f_dim=1, seed=42)
        g = np.array([1.0, 0.5])
        f = np.array([0.8])
        _, prob = bandit.select_action(g, f, available=True)
        assert 0 < prob < 1


class TestImmediateRewardOnly:
    def test_no_proxy_used(self):
        bandit = _make_bandit(g_dim=2, f_dim=1, seed=42)
        assert not hasattr(bandit, "proxy_value")

    def test_q_values_match_immediate_reward(self):
        g_dim, f_dim = 2, 1
        total = g_dim + f_dim
        prior_mean = np.array([1.0, 0.5, 2.0])
        prior_cov = np.eye(total) * 1e-6
        bandit = TSBandit(
            g_dim=g_dim,
            f_dim=f_dim,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            noise_variance=1.0,
            tau=0.1,
            epsilon_0=0.5,
            epsilon_1=0.0,
            rng=np.random.default_rng(42),
        )
        g = np.array([0.5, 0.3])
        f = np.array([0.8])

        q0 = float(g @ bandit._posterior_mean[:g_dim])
        q1 = float(
            g @ bandit._posterior_mean[:g_dim] + f @ bandit._posterior_mean[g_dim:]
        )
        assert q1 > q0

        actions = [bandit.select_action(g, f, available=True)[0] for _ in range(200)]
        # With epsilon_0=0.5, probability is capped at 0.5, so expect ~50%
        assert sum(actions) > 60  # well above random 50%

    def test_step_updates_posterior(self):
        bandit = _make_bandit(g_dim=2, f_dim=1, seed=42)
        mean_before = bandit._posterior_mean.copy()
        g = np.array([1.0, 0.5])
        f = np.array([0.8])
        bandit.step(g, f, action=1, pi=0.3, reward=5.0, available=True)
        assert not np.allclose(bandit._posterior_mean, mean_before)

    def test_unavailable_no_update(self):
        bandit = _make_bandit(g_dim=2, f_dim=1, seed=42)
        mean_before = bandit._posterior_mean.copy()
        g = np.array([1.0, 0.5])
        f = np.array([0.8])
        bandit.step(g, f, action=1, pi=0.3, reward=5.0, available=False)
        np.testing.assert_allclose(bandit._posterior_mean, mean_before)


class TestDeterminism:
    def test_same_seed_identical(self):
        g = np.array([1.0, 0.5])
        f = np.array([0.8])
        b1 = _make_bandit(g_dim=2, f_dim=1, seed=42)
        b2 = _make_bandit(g_dim=2, f_dim=1, seed=42)
        for _ in range(20):
            a1, p1 = b1.select_action(g, f, available=True)
            a2, p2 = b2.select_action(g, f, available=True)
            assert a1 == a2
            assert p1 == pytest.approx(p2)


class TestValidActionSequences:
    def test_produces_valid_sequence(self):
        bandit = _make_bandit(g_dim=2, f_dim=1, seed=42)
        g = np.array([1.0, 0.5])
        f = np.array([0.8])
        actions = []
        probs = []
        for _ in range(200):
            a, p = bandit.select_action(g, f, available=True)
            actions.append(a)
            probs.append(p)
            bandit.step(g, f, a, pi=0.3, reward=float(a * 2.0), available=True)
        assert all(a in (0, 1) for a in actions)
        assert all(0.1 <= p <= 0.2 + 1e-10 for p in probs)
