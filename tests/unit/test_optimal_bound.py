"""Tests for compute_bounds in optimal_bound.py."""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

from rl_health_interventions.config.schemas import MDPConfig, TransitionModelConfig

# The optimal_bound.py module imports from _shared at module level.
# Make the mvp directory importable so compute_bounds can be imported.
_mvp_dir = (
    pathlib.Path(__file__).resolve().parents[2] / "docs" / "experimental_phases" / "mvp"
)
sys.path.insert(0, str(_mvp_dir))

_spec = importlib.util.spec_from_file_location(
    "optimal_bound", _mvp_dir / "optimal_bound.py"
)
_optimal_bound_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_optimal_bound_mod)
compute_bounds = _optimal_bound_mod.compute_bounds


def test_returns_expected_keys(valid_config):
    bounds = compute_bounds(valid_config)
    expected = {
        "contextual_optimal",
        "noncontextual_optimal",
        "random",
        "state_names",
        "action_mats",
        "imm_reward",
        "opt_actions",
        "ctx_stationary",
        "fixed_rewards",
    }
    assert set(bounds.keys()) == expected
    assert isinstance(bounds["contextual_optimal"], float)
    assert isinstance(bounds["noncontextual_optimal"], float)
    assert isinstance(bounds["random"], float)
    assert isinstance(bounds["state_names"], list)
    assert isinstance(bounds["action_mats"], dict)
    assert isinstance(bounds["imm_reward"], dict)
    assert isinstance(bounds["opt_actions"], list)
    assert isinstance(bounds["ctx_stationary"], np.ndarray)
    assert isinstance(bounds["fixed_rewards"], dict)


def test_compute_bounds_raises_for_empty_states():
    config = MDPConfig.model_construct(
        episode_days=1,
        steps_per_day=1,
        seed=42,
        initial_state="sedentary",
        states={},
        actions=["nudge"],
        transition_model=TransitionModelConfig(
            type="rule_based",
            transition_probabilities=None,
        ),
    )
    with pytest.raises(ValueError, match="config.states is required"):
        compute_bounds(config)


def test_compute_bounds_raises_for_empty_actions():
    config = MDPConfig.model_construct(
        episode_days=1,
        steps_per_day=1,
        seed=42,
        initial_state="sedentary",
        states={"sedentary": {"reward": 0.0}},
        actions=[],
        transition_model=TransitionModelConfig(
            type="rule_based",
            transition_probabilities=None,
        ),
    )
    with pytest.raises(ValueError, match="config.actions is required"):
        compute_bounds(config)


def test_compute_bounds_raises_for_missing_transition_probabilities():
    config = MDPConfig.model_construct(
        episode_days=1,
        steps_per_day=1,
        seed=42,
        initial_state="sedentary",
        states={"sedentary": {"reward": 0.0}},
        actions=["nudge"],
        transition_model=TransitionModelConfig(
            type="rule_based",
            transition_probabilities=None,
        ),
    )
    with pytest.raises(ValueError, match="transition_probabilities is required"):
        compute_bounds(config)


def test_symmetric_two_state_mdp():
    """2-state MDP where both actions are identical 50/50 transitions.

    Stationary dist = [0.5, 0.5]; E[r] per step = 0.5 * 0 + 0.5 * 1 = 0.5.
    All bounds should be the same since actions are identical.
    """
    config = MDPConfig(
        episode_days=1,
        steps_per_day=1,
        seed=42,
        initial_state="sedentary",
        states={
            "sedentary": {"reward": 0.0},
            "active": {"reward": 1.0},
        },
        actions=["nudge", "idle"],
        transition_model={
            "type": "rule_based",
            "transition_probabilities": {
                "sedentary": {
                    "nudge": {"active": 0.5, "sedentary": 0.5},
                    "idle": {"active": 0.5, "sedentary": 0.5},
                },
                "active": {
                    "nudge": {"active": 0.5, "sedentary": 0.5},
                    "idle": {"active": 0.5, "sedentary": 0.5},
                },
            },
        },
        agents=[],
    )
    bounds = compute_bounds(config)
    assert bounds["contextual_optimal"] == pytest.approx(0.5)
    assert bounds["noncontextual_optimal"] == pytest.approx(0.5)
    assert bounds["random"] == pytest.approx(0.5)
