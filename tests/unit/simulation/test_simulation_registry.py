from __future__ import annotations

import pytest

from rl_health_interventions.simulation import REGISTRY, make
from rl_health_interventions.simulation.rule_based import RuleBasedResponse


def test_registry_populated() -> None:
    assert "rule_based" in REGISTRY
    assert REGISTRY["rule_based"] is RuleBasedResponse


def test_make_returns_instance() -> None:
    instance = make("rule_based")
    assert isinstance(instance, RuleBasedResponse)


def test_make_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="NonExistent"):
        make("NonExistent")
