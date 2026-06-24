"""Tests for HeartSteps agent, reward handler, and transition model."""

from __future__ import annotations

import numpy as np

from rl_health_interventions.agents.heartsteps import HeartStepsAgent
from rl_health_interventions.rewards.heartsteps import HeartStepsReward
from rl_health_interventions.transitions.heartsteps import HeartStepsTransition
from rl_health_interventions.state import StateView


def _make_step_data(n_days: int = 10, n_windows: int = 10) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.poisson(500, size=(1, n_days, n_windows)).astype(float)


def _make_prior(g_dim: int = 6, f_dim: int = 4) -> tuple[np.ndarray, np.ndarray]:
    total = g_dim + 2 * f_dim
    return np.zeros(total), np.eye(total)


class TestHeartStepsAgent:
    def test_select_action_returns_valid_string(self) -> None:
        step_data = _make_step_data()
        pm, pc = _make_prior()
        agent = HeartStepsAgent(
            step_data=step_data, prior_mean=pm, prior_cov=pc, seed=42
        )
        state = StateView(activity="available", day=0, step_of_day=1, steps_per_day=10)
        action = agent.select_action(state)
        assert action in ("suggest", "idle")

    def test_update_does_not_crash(self) -> None:
        step_data = _make_step_data()
        pm, pc = _make_prior()
        agent = HeartStepsAgent(
            step_data=step_data, prior_mean=pm, prior_cov=pc, seed=42
        )
        state = StateView(activity="available", day=0, step_of_day=1, steps_per_day=10)
        action = agent.select_action(state)
        agent.update(state, action, 1.0, state)

    def test_night_window_returns_idle(self) -> None:
        step_data = _make_step_data()
        pm, pc = _make_prior()
        agent = HeartStepsAgent(
            step_data=step_data, prior_mean=pm, prior_cov=pc, seed=42
        )
        state = StateView(activity="available", day=0, step_of_day=0, steps_per_day=10)
        action = agent.select_action(state)
        assert action == "idle"


class TestHeartStepsReward:
    def test_reward_returns_float(self) -> None:
        step_data = _make_step_data()
        alpha = np.array([1.0, 0.5, 0.3, 0.1, 0.2, 0.1, 0.2, 0.15, 0.1, 0.05])
        beta = np.array([1.5, 2.5, 1.0, 0.5])
        r = HeartStepsReward(step_data=step_data, alpha=alpha, beta=beta, seed=42)
        reward, done = r.reward("available", "suggest", 1)
        assert isinstance(reward, float)
        assert done is False

    def test_reward_resets_daily_steps(self) -> None:
        step_data = _make_step_data()
        alpha = np.array([1.0, 0.5, 0.3, 0.1, 0.2, 0.1, 0.2, 0.15, 0.1, 0.05])
        beta = np.array([1.5, 2.5, 1.0, 0.5])
        r = HeartStepsReward(step_data=step_data, alpha=alpha, beta=beta, seed=42)
        for i in range(10):
            r.reward("available", "idle", i)
        r1, _ = r.reward("available", "suggest", 0)
        assert isinstance(r1, float)


class TestHeartStepsTransition:
    def test_returns_valid_state(self) -> None:
        t = HeartStepsTransition(p_avail=0.85, seed=42)
        result = t.transition("available", "suggest")
        assert result in ("available", "unavailable")

    def test_night_window_unavailable(self) -> None:
        t = HeartStepsTransition(p_avail=1.0, seed=42)
        for window in range(10):
            result = t.transition("available", "suggest")
            if window in (0, 9):
                assert result == "unavailable"
