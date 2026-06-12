"""Tests for generative model (Task 2.2).

Tests trajectory generation, treatment effects, and deterministic behaviour.
"""

import numpy as np
import pytest

from paper_reproduction.data.generative_model import GenerativeModel
from paper_reproduction.data.nhanes_loader import SyntheticNHANESGenerator


@pytest.fixture
def step_data():
    gen = SyntheticNHANESGenerator(seed=42)
    return gen.generate(n_participants=10, n_days=7, n_windows=10)


@pytest.fixture
def model(step_data):
    alpha = np.array(
        [
            1.0,
            0.5,
            0.3,
            0.1,
            0.2,
            0.1,
            0.05,
            0.05,  # g features (8)
            0.2,
            0.15,
            0.1,
            0.05,  # f features (alpha_1, 4)
        ]
    )
    beta = np.array([0.5, 0.3, 0.2, 0.1])  # treatment effect (4)
    return GenerativeModel(
        step_data=step_data,
        alpha=alpha,
        beta=beta,
        noise_variance=0.5,
        p_avail=0.85,
        p_sed=0.2,
        seed=42,
    )


class TestTrajectoryGeneration:
    def test_returns_dict(self, model):
        traj = model.generate_trajectory(0)
        assert isinstance(traj, dict)

    def test_correct_keys(self, model):
        traj = model.generate_trajectory(0)
        expected_keys = {"steps", "availabilities", "actions", "rewards", "times"}
        assert set(traj.keys()) == expected_keys

    def test_trajectory_length(self, model):
        traj = model.generate_trajectory(0, n_days=90)
        n_expected = 90 * 10  # 90 days x 10 windows
        assert len(traj["steps"]) == n_expected
        assert len(traj["rewards"]) == n_expected
        assert len(traj["actions"]) == n_expected


class TestStepCountProperties:
    def test_non_negative(self, model):
        traj = model.generate_trajectory(0, 90)
        assert np.all(traj["steps"] >= 0)


class TestRewardProperties:
    def test_finite(self, model):
        traj = model.generate_trajectory(0, 90)
        assert np.all(np.isfinite(traj["rewards"]))


class TestDeterminism:
    def test_same_seed_identical(self, model):
        t1 = model.generate_trajectory(0, 90)
        m2 = GenerativeModel(
            step_data=model.step_data,
            alpha=model.alpha,
            beta=model.beta,
            noise_variance=model.noise_variance,
            p_avail=model.p_avail,
            p_sed=model.p_sed,
            seed=42,
        )
        t2 = m2.generate_trajectory(0, 90)
        np.testing.assert_array_equal(t1["steps"], t2["steps"])
        np.testing.assert_array_equal(t1["rewards"], t2["rewards"])
        np.testing.assert_array_equal(t1["actions"], t2["actions"])
