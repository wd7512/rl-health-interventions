"""Tests for the Thompson Sampling agent (Paper Section 5.3).

The agent selects actions using Thompson Sampling with probability clipping.
Action probabilities are clipped to [epsilon_1, epsilon_0] where
epsilon_0 = 0.2 (max) and epsilon_1 = 0.1 (min) to ensure bounded
exploration.

Reference:
    Liao et al. (2019). arXiv:1909.03539, Section 5.3, Equation 2.
"""

import numpy as np
import pytest

from paper_reproduction.heartsteps.agent import ThompsonSamplingAgent
from paper_reproduction.heartsteps.bayesian_regression import BayesianRewardModel
from paper_reproduction.heartsteps.dosage import DosageTracker
from paper_reproduction.heartsteps.proxy_value import ProxyValueFunction


def make_default_components(rng: np.random.Generator | None = None):
    """Create default model, dosage tracker, and proxy for tests."""
    g_dim, f_dim = 2, 1
    total = g_dim + 2 * f_dim
    model = BayesianRewardModel(
        g_dim=g_dim,
        f_dim=f_dim,
        prior_mean=np.zeros(total),
        prior_cov=np.eye(total),
        noise_variance=1.0,
    )
    dosage = DosageTracker(decay=0.95)
    proxy = ProxyValueFunction(decay=0.95, gamma=0.9, p_avail=0.8, p_sed=0.2)
    proxy.solve()
    return model, dosage, proxy


class TestActionSelection:
    def test_action_is_binary(self):
        """Selected action is always 0 or 1."""
        rng = np.random.default_rng(42)
        model, dosage, proxy = make_default_components(rng)
        agent = ThompsonSamplingAgent(
            model=model, dosage_tracker=dosage, proxy_value=proxy, rng=rng,
        )
        g = np.array([1.0, 0.5])
        f = np.array([0.8])
        for _ in range(50):
            action, _ = agent.select_action(g, f, pi=0.3, available=True)
            assert action in (0, 1)

    def test_unavailable_forces_action_zero(self):
        """When unavailable, action is always 0."""
        rng = np.random.default_rng(42)
        model, dosage, proxy = make_default_components(rng)
        agent = ThompsonSamplingAgent(
            model=model, dosage_tracker=dosage, proxy_value=proxy, rng=rng,
        )
        g = np.array([1.0, 0.5])
        f = np.array([0.8])
        for _ in range(50):
            action, _ = agent.select_action(g, f, pi=0.3, available=False)
            assert action == 0

    def test_probability_clipping(self):
        """Action probabilities are clipped to [epsilon_1, epsilon_0]."""
        rng = np.random.default_rng(42)
        model, dosage, proxy = make_default_components(rng)
        agent = ThompsonSamplingAgent(
            model=model,
            dosage_tracker=dosage,
            proxy_value=proxy,
            epsilon_0=0.2,
            epsilon_1=0.1,
            rng=rng,
        )
        g = np.array([1.0, 0.5])
        f = np.array([0.8])
        for _ in range(50):
            _, prob = agent.select_action(g, f, pi=0.3, available=True)
            assert 0.1 <= prob <= 0.2 + 1e-10

    def test_action_variation(self):
        """Multiple calls with same state produce varied actions (stochastic)."""
        rng = np.random.default_rng(42)
        model, dosage, proxy = make_default_components(rng)
        agent = ThompsonSamplingAgent(
            model=model, dosage_tracker=dosage, proxy_value=proxy, rng=rng,
        )
        g = np.array([1.0, 0.5])
        f = np.array([0.8])
        actions = set()
        for _ in range(100):
            action, _ = agent.select_action(g, f, pi=0.3, available=True)
            actions.add(action)
        # With 100 draws and clipping, both actions should appear
        assert len(actions) == 2

    def test_posterior_update(self):
        """Agent updates model posterior after step()."""
        rng = np.random.default_rng(42)
        model, dosage, proxy = make_default_components(rng)
        mean_before = model.posterior_mean.copy()
        agent = ThompsonSamplingAgent(
            model=model, dosage_tracker=dosage, proxy_value=proxy, rng=rng,
        )
        g = np.array([1.0, 0.5])
        f = np.array([0.8])
        agent.step(
            g=g, f=f, action=1, pi=0.3, reward=5.0, available=True,
        )
        # Posterior should have changed
        assert not np.allclose(model.posterior_mean, mean_before)

    def test_dosage_tracker_updated(self):
        """Agent updates dosage tracker after step()."""
        rng = np.random.default_rng(42)
        model, dosage, proxy = make_default_components(rng)
        agent = ThompsonSamplingAgent(
            model=model, dosage_tracker=dosage, proxy_value=proxy, rng=rng,
        )
        dosage_before = dosage.value
        g = np.array([1.0, 0.5])
        f = np.array([0.8])
        agent.step(
            g=g, f=f, action=1, pi=0.3, reward=5.0, available=True,
        )
        # Dosage should have changed (treatment delivered)
        assert dosage.value != dosage_before

    def test_select_action_returns_probability(self):
        """select_action returns a probability in (0, 1)."""
        rng = np.random.default_rng(42)
        model, dosage, proxy = make_default_components(rng)
        agent = ThompsonSamplingAgent(
            model=model, dosage_tracker=dosage, proxy_value=proxy, rng=rng,
        )
        g = np.array([1.0, 0.5])
        f = np.array([0.8])
        _, prob = agent.select_action(g, f, pi=0.3, available=True)
        assert 0 < prob < 1
