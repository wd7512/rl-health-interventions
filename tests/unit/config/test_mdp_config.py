from __future__ import annotations

from pathlib import Path

import pydantic
import pytest
import yaml

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.config.schemas import MDPConfig

ASSETS = Path(__file__).parent.parent.parent / "assets"


def test_mdp_config_loaded_from_yaml():
    path = ASSETS / "valid_mdp_config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = MDPConfig.model_validate(raw)
    assert config.steps_per_day == 5
    assert config.episode_days == 90
    assert config.initial_state == "sedentary"
    assert config.seed == 42
    assert set(config.states) == {"sedentary", "active"}
    assert config.actions == ["nudge", "idle"]
    assert config.transition_model.type == "rule_based"
    assert len(config.agents) == 1
    assert config.agents[0].type == "thompson_sampling"


def test_mdp_config_precomputes_per_step_reward():
    path = ASSETS / "valid_mdp_config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = MDPConfig.model_validate(raw)
    assert config.per_step_reward is not None
    assert len(config.per_step_reward) == 5
    for step_reward in config.per_step_reward:
        assert step_reward["sedentary"] == 0.0
        assert step_reward["active"] == 1.0


def test_load_config_reads_yaml_file():
    path = ASSETS / "valid_mdp_config.yaml"
    config = load_config(path)
    assert isinstance(config, MDPConfig)
    assert config.steps_per_day == 5


def test_mdp_config_rejects_missing_transition_model():
    path = ASSETS / "valid_mdp_config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    del raw["transition_model"]
    with pytest.raises(pydantic.ValidationError):
        MDPConfig.model_validate(raw)


def test_mdp_config_rejects_negative_episode_days():
    path = ASSETS / "valid_mdp_config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["episode_days"] = -1
    with pytest.raises(pydantic.ValidationError):
        MDPConfig.model_validate(raw)


def test_mdp_config_with_reward_multiplier():
    path = ASSETS / "valid_mdp_config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["reward_multiplier_by_step"] = [1, 1, 1, 1, 0]
    config = MDPConfig.model_validate(raw)
    assert config.per_step_reward is not None
    assert config.per_step_reward[4]["active"] == 0.0
    assert config.per_step_reward[0]["active"] == 1.0
