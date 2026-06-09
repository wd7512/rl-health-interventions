from __future__ import annotations

from rl_health_interventions.rewards import REGISTRY, make
from rl_health_interventions.rewards.compound import CompoundReward


def test_registry_populated() -> None:
    assert "CompoundReward" in REGISTRY
    assert REGISTRY["CompoundReward"] is CompoundReward


def test_make_returns_instance() -> None:
    instance = make("CompoundReward")
    assert isinstance(instance, CompoundReward)


def test_make_unknown_raises_keyerror() -> None:
    try:
        make("NonExistent")
        assert False, "Expected KeyError"
    except KeyError as e:
        assert "NonExistent" in str(e)
        assert "CompoundReward" in str(e)
