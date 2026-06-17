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


def test_make_with_config_as_first_arg(minimal_config) -> None:
    """make() accepts config as positional first argument."""
    instance = make(minimal_config)
    assert isinstance(instance, CompoundReward)


def test_make_without_config_or_name_raises() -> None:
    """make() raises TypeError when neither config nor name is given."""
    with pytest.raises(TypeError):
        make()
