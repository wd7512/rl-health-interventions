from __future__ import annotations

import pytest

from rl_health_interventions.transitions import REGISTRY, make
from rl_health_interventions.transitions.rule_based import RuleBasedTransition


def test_registry_populated() -> None:
    assert "rule_based" in REGISTRY
    assert REGISTRY["rule_based"] is RuleBasedTransition


def test_make_returns_instance() -> None:
    from rl_health_interventions.config.schemas import (
        ActivityLevel,
        Action,
        TimeOfDay,
        MDPConfig,
        TransitionMatrix,
        TimeOfDayMask,
    )

    config = MDPConfig(
        activity_levels=[ActivityLevel.SEDENTARY, ActivityLevel.ACTIVE],
        actions=[Action.SEND, Action.DON_T_SEND],
        time_of_day=[TimeOfDay.MORNING],
        steps_per_day=1,
        transition=TransitionMatrix(
            root={
                ActivityLevel.SEDENTARY: {
                    Action.SEND: {
                        ActivityLevel.SEDENTARY: 0.5,
                        ActivityLevel.ACTIVE: 0.5,
                    },
                    Action.DON_T_SEND: {
                        ActivityLevel.SEDENTARY: 0.5,
                        ActivityLevel.ACTIVE: 0.5,
                    },
                },
                ActivityLevel.ACTIVE: {
                    Action.SEND: {
                        ActivityLevel.SEDENTARY: 0.5,
                        ActivityLevel.ACTIVE: 0.5,
                    },
                    Action.DON_T_SEND: {
                        ActivityLevel.SEDENTARY: 0.5,
                        ActivityLevel.ACTIVE: 0.5,
                    },
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
    instance = make("rule_based", config=config)
    assert isinstance(instance, RuleBasedTransition)


def test_make_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="NonExistent"):
        make("NonExistent")
