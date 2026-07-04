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
    assert config.initial_state == {"activity_level": "sedentary"}
    assert config.seed == 42
    assert "activity_level" in config.state.variables
    assert config.action_names == ["nudge", "idle"]
    assert config.transition_model.type == "rule_based"
    assert len(config.agents) == 1
    assert config.agents[0].type == "thompson_sampling"


def test_load_config_reads_yaml_file():
    path = ASSETS / "valid_mdp_config.yaml"
    config = load_config(path)
    assert isinstance(config, MDPConfig)
    assert config.steps_per_day == 5


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.pop("transition_model"),
        lambda raw: raw.update({"episode_days": -1}),
    ],
)
def test_mdp_config_rejects_invalid(mutate):
    path = ASSETS / "valid_mdp_config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(raw)
    with pytest.raises(Exception):
        MDPConfig.model_validate(raw)
