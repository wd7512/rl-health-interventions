"""Integration tests for the unified MDPConfig model."""

import pytest
from pydantic import ValidationError

from rl_health_interventions.config.schemas import MDPConfig


def test_valid_minimal_config():
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
    )
    assert config.action_names == ["nudge", "idle"]


def test_reward_with_action_variable():
    config = MDPConfig(
        episode_days=1,
        steps_per_day=1,
        seed=42,
        state={"variables": {"activity_level": {"names": ["sedentary", "active"]}}},
        initial_state={"activity_level": "sedentary"},
        actions={"nudge": {}, "idle": {}},
        reward={
            "variables": {
                "value": {
                    "source": "state.activity_level",
                    "mapping": {"sedentary": 0.0, "active": 1.0},
                },
                "penalty": {
                    "source": "action",
                    "mapping": {"nudge": 0.05, "idle": 0.0},
                },
            },
            "formula": "value - penalty",
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
    )
    assert config.action_names == ["nudge", "idle"]
    assert "penalty" in config.reward.variables


def test_reward_missing_action_in_mapping_rejected():
    with pytest.raises(ValidationError, match="not in mapping"):
        MDPConfig(
            episode_days=1,
            steps_per_day=1,
            state={"variables": {"activity_level": {"names": ["sedentary", "active"]}}},
            initial_state={"activity_level": "sedentary"},
            actions=["nudge", "idle"],
            reward={
                "variables": {
                    "penalty": {
                        "source": "action",
                        "mapping": {"nudge": 0.05},
                    }
                },
                "formula": "penalty",
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
        )
