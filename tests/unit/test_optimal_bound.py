"""Tests for compute_bounds in optimal_bound.py."""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

from rl_health_interventions.config.schemas import MDPConfig

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

_HERE = pathlib.Path(__file__).resolve().parent.parent
ASSETS_TABLES = str(_HERE / "assets" / "tables")


def _symmetric_config() -> MDPConfig:
    """Config with 50/50 symmetric transitions via minimal.json table."""
    return MDPConfig(
        episode_days=1,
        steps_per_day=1,
        seed=42,
        initial_state={"activity_level": "sedentary"},
        state={
            "factors": {
                "activity_level": {"dims": 2, "names": ["sedentary", "active"]},
            },
        },
        actions=["nudge", "idle"],
        reward={
            "factor": "activity_level",
            "values": {"sedentary": 0.0, "active": 1.0},
        },
        transition_model={"type": "rule_based", "table_dir": ASSETS_TABLES},
    )


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


def test_compute_bounds_raises_for_schema_reference():
    config = MDPConfig(
        episode_days=1,
        steps_per_day=1,
        seed=42,
        initial_state={"activity_level": "sedentary"},
        state={"schema": "some_schema_ref"},
        actions=["nudge", "idle"],
        reward={
            "factor": "activity_level",
            "values": {"sedentary": 0.0, "active": 1.0},
        },
        transition_model={"type": "learned"},
    )
    with pytest.raises(ValueError, match="actual state definitions"):
        compute_bounds(config)


def test_symmetric_two_state_mdp():
    """2-state MDP using mvp_rule_based.json transitions."""
    config = _symmetric_config()
    bounds = compute_bounds(config)
    assert bounds["contextual_optimal"] == pytest.approx(3.0 / 7.0)
    assert bounds["noncontextual_optimal"] == pytest.approx(3.0 / 8.0)
    assert bounds["random"] == pytest.approx(4.0 / 13.0)


def test_compute_bounds_raises_for_invalid_state_count():
    with pytest.raises(ValueError, match="exactly 2 states"):
        config = MDPConfig(
            episode_days=1,
            steps_per_day=1,
            seed=42,
            initial_state={"activity_level": "a"},
            state={
                "factors": {
                    "activity_level": {"dims": 3, "names": ["a", "b", "c"]},
                },
            },
            actions=["nudge"],
            reward={
                "factor": "activity_level",
                "values": {"a": 0.0, "b": 0.5, "c": 1.0},
            },
            transition_model={"type": "rule_based", "table_dir": ASSETS_TABLES},
        )
        compute_bounds(config)
