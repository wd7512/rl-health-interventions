from __future__ import annotations

import pytest

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.transitions import REGISTRY, make
from rl_health_interventions.transitions.random import RandomTransition
from rl_health_interventions.transitions.rule_based import RuleBasedTransition


def test_registry_populated() -> None:
    assert "rule_based" in REGISTRY
    assert REGISTRY["rule_based"] is RuleBasedTransition
    assert "random" in REGISTRY
    assert REGISTRY["random"] is RandomTransition


def test_make_returns_instance(valid_config) -> None:
    instance = make("rule_based", config=valid_config)
    assert isinstance(instance, RuleBasedTransition)


def test_make_random_returns_instance() -> None:
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
        transition_model={"type": "random"},
        agents=[],
    )
    instance = make("random", config=config)
    assert isinstance(instance, RandomTransition)


def test_make_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="NonExistent"):
        make("NonExistent")


def test_make_with_config_as_first_arg(valid_config) -> None:
    """make() accepts config as positional first argument."""
    instance = make(valid_config)
    assert isinstance(instance, RuleBasedTransition)


def test_make_without_config_or_name_raises() -> None:
    """make() raises TypeError when neither config nor name is given."""
    with pytest.raises(TypeError):
        make()
