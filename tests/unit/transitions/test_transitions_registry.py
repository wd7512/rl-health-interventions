from __future__ import annotations

import pytest

from rl_health_interventions.transitions import REGISTRY, make
from rl_health_interventions.transitions.random import RandomTransition
from rl_health_interventions.transitions.rule_based import RuleBasedTransition


def test_registry_populated() -> None:
    assert "rule_based" in REGISTRY
    assert REGISTRY["rule_based"] is RuleBasedTransition
    assert "random" in REGISTRY
    assert REGISTRY["random"] is RandomTransition
    assert "bootstrap" in REGISTRY
    assert REGISTRY["bootstrap"] is RuleBasedTransition


def test_make_returns_instance(valid_config) -> None:
    instance = make("rule_based", config=valid_config)
    assert isinstance(instance, RuleBasedTransition)


def test_make_random_returns_instance(valid_config) -> None:
    instance = make("random", config=valid_config)
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
