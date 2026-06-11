from __future__ import annotations

import pytest

from rl_health_interventions.transitions import REGISTRY, make
from rl_health_interventions.transitions.rule_based import RuleBasedTransition


def test_registry_populated() -> None:
    assert "rule_based" in REGISTRY
    assert REGISTRY["rule_based"] is RuleBasedTransition


def test_make_returns_instance() -> None:
    instance = make("rule_based")
    assert isinstance(instance, RuleBasedTransition)


def test_make_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="NonExistent"):
        make("NonExistent")
