"""Tests for compute_bounds in optimal_bound.py."""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

from rl_health_interventions.config.schemas import (
    MDPConfig,
    RewardConfig,
    StateConfig,
    TransitionModelConfig,
    TransitionProbabilities,
)

_mvp_dir = (
    pathlib.Path(__file__).resolve().parents[2] / "docs" / "experimental_phases" / "mvp"
)
sys.path.insert(0, str(_mvp_dir))
try:
    _spec = importlib.util.spec_from_file_location(
        "optimal_bound", _mvp_dir / "optimal_bound.py"
    )
    assert _spec is not None, "Failed to load module spec"
    assert _spec.loader is not None, "Module spec has no loader"
    _optimal_bound_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_optimal_bound_mod)
finally:
    if sys.path and sys.path[0] == str(_mvp_dir):
        sys.path.pop(0)
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


def test_compute_bounds_raises_for_empty_variables():
    config = MDPConfig.model_construct(
        episode_days=1,
        steps_per_day=1,
        seed=42,
        state=StateConfig.model_validate({"variables": {}}),
        initial_state={},
        actions={},
        reward=RewardConfig.model_validate({"variables": {}, "formula": "0"}),
        transition_model=TransitionModelConfig(
            type="rule_based",
            transition_probabilities=None,
        ),
    )
    with pytest.raises(ValueError, match="variables is required"):
        compute_bounds(config)


def test_compute_bounds_raises_for_missing_transition_probabilities():
    config = MDPConfig.model_construct(
        episode_days=1,
        steps_per_day=1,
        seed=42,
        state=StateConfig.model_validate(
            {
                "variables": {"activity_level": {"names": ["sedentary", "active"]}},
            }
        ),
        initial_state={"activity_level": "sedentary"},
        actions={"nudge": {}, "idle": {}},
        reward=RewardConfig.model_validate(
            {
                "variables": {
                    "value": {
                        "source": "state.activity_level",
                        "mapping": {"sedentary": 0.0, "active": 1.0},
                    }
                },
                "formula": "value",
            }
        ),
        transition_model=TransitionModelConfig(
            type="rule_based",
            transition_probabilities=None,
        ),
    )
    with pytest.raises(ValueError, match="transition_probabilities is required"):
        compute_bounds(config)


def test_symmetric_two_state_mdp():
    config = MDPConfig(
        episode_days=1,
        steps_per_day=1,
        seed=42,
        state={"variables": {"activity_level": {"names": ["sedentary", "active"]}}},
        initial_state={"activity_level": "sedentary"},
        actions=["nudge", "idle"],
        reward={
            "variables": {
                "value": {
                    "source": "state.activity_level",
                    "mapping": {"sedentary": 0.0, "active": 1.0},
                }
            },
            "formula": "value",
        },
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


def test_compute_bounds_raises_for_invalid_state_count():
    config = MDPConfig(
        episode_days=1,
        steps_per_day=1,
        seed=42,
        state={"variables": {"category": {"names": ["a", "b", "c"]}}},
        initial_state={"category": "a"},
        actions=["nudge"],
        reward={
            "variables": {
                "value": {
                    "source": "state.category",
                    "mapping": {"a": 0.0, "b": 0.5, "c": 1.0},
                }
            },
            "formula": "value",
        },
        transition_model=TransitionModelConfig(
            type="rule_based",
            transition_probabilities=TransitionProbabilities(
                {
                    "a": {"nudge": {"a": 0.3, "b": 0.4, "c": 0.3}},
                    "b": {"nudge": {"a": 0.3, "b": 0.4, "c": 0.3}},
                    "c": {"nudge": {"a": 0.3, "b": 0.4, "c": 0.3}},
                }
            ),
        ),
        agents=[],
    )
    with pytest.raises(ValueError, match="exactly 2 states"):
        compute_bounds(config)
