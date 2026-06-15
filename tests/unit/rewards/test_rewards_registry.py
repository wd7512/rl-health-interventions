from __future__ import annotations

import pytest

from rl_health_interventions.rewards import REGISTRY, make
from rl_health_interventions.rewards.compound import CompoundReward
from rl_health_interventions.config.schemas import (
    ActivityLevel,
    Action,
    MDPConfig,
    TransitionMatrix,
    TimeOfDayMask,
    TimeOfDay,
)


def _minimal_config() -> MDPConfig:
    return MDPConfig(
        activity_levels=[ActivityLevel.SEDENTARY, ActivityLevel.ACTIVE],
        actions=[Action.SEND, Action.DON_T_SEND],
        time_of_day=[TimeOfDay.MORNING],
        transition=TransitionMatrix(
            root={
                ActivityLevel.SEDENTARY: {
                    Action.SEND: {
                        ActivityLevel.SEDENTARY: 0.5,
                        ActivityLevel.ACTIVE: 0.5,
                    }
                },
                ActivityLevel.ACTIVE: {
                    Action.SEND: {
                        ActivityLevel.SEDENTARY: 0.5,
                        ActivityLevel.ACTIVE: 0.5,
                    }
                },
            }
        ),
        masks=TimeOfDayMask(
            root={
                TimeOfDay.MORNING: {
                    ActivityLevel.SEDENTARY: 0.0,
                    ActivityLevel.ACTIVE: 0.0,
                }
            }
        ),
    )


def test_registry_populated() -> None:
    assert "compound" in REGISTRY
    assert REGISTRY["compound"] is CompoundReward


def test_make_returns_instance() -> None:
    instance = make("compound", config=_minimal_config())
    assert isinstance(instance, CompoundReward)


def test_make_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="NonExexistent"):
        make("NonExexistent")
