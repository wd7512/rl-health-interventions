from __future__ import annotations

from rl_health_interventions.simulation import REGISTRY, make
from rl_health_interventions.simulation.rule_based import RuleBasedResponse


def test_registry_populated() -> None:
    assert "RuleBasedResponse" in REGISTRY
    assert REGISTRY["RuleBasedResponse"] is RuleBasedResponse


def test_make_returns_instance() -> None:
    instance = make("RuleBasedResponse")
    assert isinstance(instance, RuleBasedResponse)


def test_make_unknown_raises_keyerror() -> None:
    try:
        make("NonExistent")
        assert False, "Expected KeyError"
    except KeyError as e:
        assert "NonExistent" in str(e)
        assert "RuleBasedResponse" in str(e)
