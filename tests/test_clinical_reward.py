"""Tests for clinical reward module."""

import numpy as np
from src.rl_health_interventions.clinical_reward import (
    ClinicalOutcomeSimulator,
    ClinicalParams,
    compute_clinical_reward
)
from src.rl_health_interventions.reward import HealthReward


def test_clinical_simulator_reset():
    sim = ClinicalOutcomeSimulator()
    sim.add_daily_steps(8000)
    assert len(sim) == 1
    sim.reset()
    assert len(sim) == 0


def test_clinical_change_no_data():
    sim = ClinicalOutcomeSimulator()
    change = sim.compute_clinical_change()
    assert change['blood_pressure_change'] == 0.0
    assert change['hba1c_change'] == 0.0


def test_clinical_change_with_data():
    sim = ClinicalOutcomeSimulator(params=ClinicalParams(noise_std=0.0))
    # Provide 21 days of steps above threshold
    for _ in range(21):
        sim.add_daily_steps(10000.0)
    change = sim.compute_clinical_change()
    # Expect negative bp_change (improvement)
    avg_excess = (10000 - 7000) / 1000.0  # 3.0
    expected_bp = -0.5 * 3.0  # -1.5
    assert np.isclose(change['blood_pressure_change'], expected_bp, atol=1e-5)
    expected_hba1c = expected_bp * 0.2  # -0.3
    assert np.isclose(change['hba1c_change'], expected_hba1c, atol=1e-5)


def test_clinical_reward_computation():
    sim = ClinicalOutcomeSimulator(params=ClinicalParams(noise_std=0.0))
    for _ in range(21):
        sim.add_daily_steps(7000.0)  # exactly at threshold -> no change
    reward = compute_clinical_reward(sim, bp_scale=-0.5, hba1c_scale=-1.0)
    assert np.isclose(reward, 0.0, atol=1e-5)


def test_health_reward_accumulation():
    hr = HealthReward(
        step_target=7500,
        step_reward_scale=0.01,
        bp_scale=-0.5,
        hba1c_scale=-1.0,
        clinical_frequency=21
    )
    hr.reset()
    total_reward = 0.0
    for day in range(21):
        steps = 8000 if day % 2 == 0 else 6000
        daily_rew = hr(steps, done=(day == 20))
        total_reward += daily_rew
    # At day 21, clinical reward is added
    # We expect clinical reward to be computed based on average of 21 days
    # Not checking exact value due to noise, but ensure it's called.
    assert hr.episode_clinical_reward != 0.0  # Should have non-zero clinical reward


def test_health_reward_reset():
    hr = HealthReward()
    hr.reset()
    assert hr.day_count == 0
    assert hr.episode_clinical_reward == 0.0
    assert len(hr.clinical_sim) == 0
