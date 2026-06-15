from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.config.schemas import (
    MDPConfig,
    ActivityLevel,
)

VALID_YAML = """
activity_levels:
  - sedentary
  - active
actions:
  - send
  - don_t_send
time_of_day:
  - morning
  - midday
  - afternoon
  - evening
  - night
steps_per_day: 5
episode_days: 90
initial_state: sedentary
reward_active: 1.0
reward_sedentary: 0.0
seed: 42
transition:
  sedentary:
    send:
      active: 0.3
      sedentary: 0.7
    don_t_send:
      active: 0.1
      sedentary: 0.9
  active:
    send:
      active: 0.8
      sedentary: 0.2
    don_t_send:
      active: 0.6
      sedentary: 0.4
masks:
  morning:
    sedentary: 0.0
    active: 0.0
  midday:
    sedentary: 0.0
    active: 0.0
  afternoon:
    sedentary: 0.0
    active: 0.0
  evening:
    sedentary: 0.0
    active: 0.0
  night:
    sedentary: 1.0
    active: 1.0
"""


def test_mdp_config_loads_valid_yaml():
    raw = yaml.safe_load(VALID_YAML)
    config = MDPConfig.model_validate(raw)
    assert config.steps_per_day == 5
    assert config.episode_days == 90
    assert config.initial_state == ActivityLevel.SEDENTARY
    assert config.reward_active == 1.0
    assert config.reward_sedentary == 0.0
    assert config.seed == 42


def test_mdp_config_rejects_missing_transition_matrix():
    raw = yaml.safe_load(VALID_YAML)
    del raw["transition"]
    with pytest.raises(ValidationError):
        MDPConfig.model_validate(raw)


def test_mdp_config_rejects_probabilities_not_summing_to_one():
    raw = yaml.safe_load(VALID_YAML)
    raw["transition"]["sedentary"]["send"]["active"] = 0.5
    with pytest.raises(ValidationError):
        MDPConfig.model_validate(raw)


def test_mdp_config_rejects_unknown_time_of_day():
    raw = yaml.safe_load(VALID_YAML)
    raw["time_of_day"].append("midnight")
    with pytest.raises(ValidationError):
        MDPConfig.model_validate(raw)


def test_load_config_reads_yaml_file(tmp_path: Path):
    path = tmp_path / "test_config.yaml"
    path.write_text(VALID_YAML, encoding="utf-8")
    config = load_config(path)
    assert isinstance(config, MDPConfig)
    assert config.steps_per_day == 5
