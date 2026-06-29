from __future__ import annotations

from pathlib import Path

import pytest

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.config.schemas import MDPConfig

ASSETS = Path(__file__).parent.parent.parent / "assets"


def test_mdp_config_loaded_from_yaml():
    config = load_config(ASSETS / "valid_mdp_config.yaml")
    assert config.steps_per_day == 5
    assert config.episode_days == 90
    assert config.seed == 42
    assert isinstance(config.initial_state, dict)
    assert config.initial_state["activity_level"] == "sedentary"
    assert set(config.actions.keys()) == {"nudge", "idle"}
    assert config.transition_model.type == "rule_based"
    assert config.transition_model.table_dir.endswith("tests/assets/tables")
    assert len(config.agents) == 1
    assert config.agents[0].type == "thompson_sampling"


def test_mdp_config_action_names_property():
    config = load_config(ASSETS / "valid_mdp_config.yaml")
    assert sorted(config.action_names) == ["idle", "nudge"]


def test_mdp_config_factor_configs_property():
    config = load_config(ASSETS / "valid_mdp_config.yaml")
    assert "activity_level" in config.factor_configs
    assert config.factor_configs["activity_level"].dims == 2
    assert config.factor_configs["activity_level"].names == ["sedentary", "active"]


def test_actions_list_auto_conversion():
    """Actions list shorthand is auto-converted to dict."""
    raw = {
        "episode_days": 1,
        "steps_per_day": 1,
        "seed": 42,
        "initial_state": {"activity_level": "sedentary"},
        "state": {
            "factors": {
                "activity_level": {"dims": 2, "names": ["sedentary", "active"]},
            },
        },
        "actions": ["nudge", "idle"],
        "reward": {
            "factor": "activity_level",
            "values": {"sedentary": 0.0, "active": 1.0},
        },
        "transition_model": {"type": "rule_based", "table_dir": "tables"},
    }
    config = MDPConfig.model_validate(raw)
    assert isinstance(config.actions, dict)
    assert "nudge" in config.actions
    assert "idle" in config.actions
    assert config.actions["nudge"].action_penalty == 0.0
    assert config.actions["idle"].action_penalty == 0.0
    assert sorted(config.action_names) == ["idle", "nudge"]


def test_actions_dict_preserved():
    """Actions as dict is kept as-is."""
    raw = {
        "episode_days": 1,
        "steps_per_day": 1,
        "seed": 42,
        "initial_state": {"activity_level": "sedentary"},
        "state": {
            "factors": {
                "activity_level": {"dims": 2, "names": ["sedentary", "active"]},
            },
        },
        "actions": {"nudge": {"action_penalty": 0.5}, "idle": {"action_penalty": 0.0}},
        "reward": {
            "factor": "activity_level",
            "values": {"sedentary": 0.0, "active": 1.0},
        },
        "transition_model": {"type": "rule_based", "table_dir": "tables"},
    }
    config = MDPConfig.model_validate(raw)
    assert config.actions["nudge"].action_penalty == 0.5
    assert config.actions["idle"].action_penalty == 0.0


def test_load_config_reads_yaml_file():
    path = ASSETS / "valid_mdp_config.yaml"
    config = load_config(path)
    assert isinstance(config, MDPConfig)
    assert config.steps_per_day == 5


def test_minimal_config_loads():
    path = ASSETS / "minimal_config.yaml"
    config = load_config(path)
    assert isinstance(config, MDPConfig)
    assert config.episode_days == 1
    assert config.steps_per_day == 1


def test_mdp_config_rejects_negative_episode_days():
    raw = {
        "episode_days": -1,
        "steps_per_day": 1,
        "seed": 42,
        "initial_state": {"activity_level": "sedentary"},
        "state": {
            "factors": {
                "activity_level": {"dims": 2, "names": ["sedentary", "active"]},
            },
        },
        "actions": ["nudge"],
        "reward": {
            "factor": "activity_level",
            "values": {"sedentary": 0.0, "active": 1.0},
        },
        "transition_model": {"type": "rule_based", "table_dir": "tables"},
    }
    with pytest.raises(Exception):
        MDPConfig.model_validate(raw)


def test_per_step_multiplier_validation():
    """per_step_multiplier length must match steps_per_day."""
    raw = {
        "episode_days": 1,
        "steps_per_day": 5,
        "seed": 42,
        "initial_state": {"activity_level": "sedentary"},
        "state": {
            "factors": {
                "activity_level": {"dims": 2, "names": ["sedentary", "active"]},
            },
        },
        "actions": ["nudge"],
        "reward": {
            "factor": "activity_level",
            "values": {"sedentary": 0.0, "active": 1.0},
            "per_step_multiplier": [1, 2, 3],  # wrong length
        },
        "transition_model": {"type": "rule_based", "table_dir": "tables"},
    }
    with pytest.raises(Exception, match="per_step_multiplier|length"):
        MDPConfig.model_validate(raw)


def test_reward_factor_must_exist_in_state_factors():
    raw = {
        "episode_days": 1,
        "steps_per_day": 1,
        "seed": 42,
        "initial_state": {"activity_level": "sedentary"},
        "state": {
            "factors": {
                "activity_level": {"dims": 2, "names": ["sedentary", "active"]},
            },
        },
        "actions": ["nudge"],
        "reward": {
            "factor": "nonexistent_factor",
            "values": {"x": 0.0},
        },
        "transition_model": {"type": "rule_based", "table_dir": "tables"},
    }
    with pytest.raises(Exception, match="factor.*nonexistent_factor"):
        MDPConfig.model_validate(raw)


def test_reward_values_keys_must_match_factor_names():
    raw = {
        "episode_days": 1,
        "steps_per_day": 1,
        "seed": 42,
        "initial_state": {"activity_level": "sedentary"},
        "state": {
            "factors": {
                "activity_level": {"dims": 2, "names": ["sedentary", "active"]},
            },
        },
        "actions": ["nudge"],
        "reward": {
            "factor": "activity_level",
            "values": {"wrong_key": 0.0, "active": 1.0},
        },
        "transition_model": {"type": "rule_based", "table_dir": "tables"},
    }
    with pytest.raises(Exception, match="values"):
        MDPConfig.model_validate(raw)


def test_initial_state_keys_must_match_factor_names():
    raw = {
        "episode_days": 1,
        "steps_per_day": 1,
        "seed": 42,
        "initial_state": {"wrong_factor": "sedentary"},
        "state": {
            "factors": {
                "activity_level": {"dims": 2, "names": ["sedentary", "active"]},
            },
        },
        "actions": ["nudge"],
        "reward": {
            "factor": "activity_level",
            "values": {"sedentary": 0.0, "active": 1.0},
        },
        "transition_model": {"type": "rule_based", "table_dir": "tables"},
    }
    with pytest.raises(Exception, match="initial_state|factor"):
        MDPConfig.model_validate(raw)


def test_initial_state_values_must_be_valid_factor_values():
    raw = {
        "episode_days": 1,
        "steps_per_day": 1,
        "seed": 42,
        "initial_state": {"activity_level": "nonexistent_value"},
        "state": {
            "factors": {
                "activity_level": {"dims": 2, "names": ["sedentary", "active"]},
            },
        },
        "actions": ["nudge"],
        "reward": {
            "factor": "activity_level",
            "values": {"sedentary": 0.0, "active": 1.0},
        },
        "transition_model": {"type": "rule_based", "table_dir": "tables"},
    }
    with pytest.raises(Exception, match="initial_state|nonexistent"):
        MDPConfig.model_validate(raw)


def test_schema_ref_mode_skips_cross_references():
    """When state has schema, cross-referencing validators are skipped."""
    raw = {
        "episode_days": 1,
        "steps_per_day": 1,
        "seed": 42,
        "initial_state": {"activity_level": "sedentary"},
        "state": {"schema": "heartsteps"},
        "actions": {"schema": "heartsteps"},
        "reward": {
            "factor": "activity_level",
            "values": {"sedentary": 0.0, "active": 1.0},
        },
        "transition_model": {"type": "learned"},
    }
    config = MDPConfig.model_validate(raw)
    assert config.state.schema == "heartsteps"


def test_sprint1_config_loads():
    """Load the Sprint 1 config from docs location."""
    path = (
        Path(__file__).parent.parent.parent.parent
        / "docs"
        / "sprint1"
        / "configs"
        / "sprint1.yaml"
    )
    if not path.exists():
        pytest.skip("sprint1.yaml not found")
    config = load_config(path)
    assert isinstance(config, MDPConfig)
    assert len(config.factor_configs) == 4
    assert "step_bin" in config.factor_configs
    assert config.factor_configs["step_bin"].boundaries == [800, 1600]
    assert "burden" in config.factor_configs
    assert config.factor_configs["burden"].dims == 3
    assert sorted(config.action_names) == [
        "goal_reminder",
        "idle",
        "journal",
        "movement_suggestion",
    ]
    assert config.actions["idle"].action_penalty == 0
    assert config.actions["movement_suggestion"].action_penalty == 1
    assert config.reward.factor == "step_bin"
    assert config.reward.values == {"inactive": 0.0, "moderate": 0.5, "active": 1.0}
    assert config.reward.action_penalty_multiplier == 0.05
    assert config.initial_state == {
        "step_bin": "inactive",
        "sleep": "rested",
        "day_of_week": "weekday",
        "burden": "low",
    }
