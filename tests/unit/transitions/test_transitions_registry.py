from __future__ import annotations

from rl_health_interventions.transitions import REGISTRY, make
from rl_health_interventions.transitions.rule_based import RuleBasedTransition


def test_registry_populated() -> None:
    assert "RuleBasedTransition" in REGISTRY
    assert REGISTRY["RuleBasedTransition"] is RuleBasedTransition


def test_make_returns_instance() -> None:
    instance = make("RuleBasedTransition")
    assert isinstance(instance, RuleBasedTransition)


def test_make_unknown_raises_keyerror() -> None:
    try:
        make("NonExistent")
        assert False, "Expected KeyError"
    except KeyError as e:
        assert "NonExistent" in str(e)
        assert "RuleBasedTransition" in str(e)
