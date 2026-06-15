from __future__ import annotations

import pytest

from rl_health_interventions.rewards import REGISTRY, make
from rl_health_interventions.rewards.compound import CompoundReward


def test_registry_populated() -> None:
    assert "compound" in REGISTRY
    assert REGISTRY["compound"] is CompoundReward


def test_make_returns_instance(minimal_config) -> None:
    instance = make("compound", config=minimal_config)
    assert isinstance(instance, CompoundReward)


def test_make_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="NonExistent"):
        make("NonExistent")
