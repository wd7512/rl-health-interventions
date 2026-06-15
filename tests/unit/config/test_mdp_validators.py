import pytest
from pydantic import ValidationError

from rl_health_interventions.config.schemas import (
    MDPConfig,
    ActivityLevel,
    Action,
    TimeOfDay,
    TransitionMatrix,
    TimeOfDayMask,
)


def _make_config(**overrides):
    defaults = dict(
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
                },
            }
        ),
    )
    defaults.update(overrides)
    return MDPConfig(**defaults)


def test_negative_probability_rejected():
    """Negative probability caught at TransitionMatrix level."""
    with pytest.raises(ValidationError, match="cannot be negative"):
        TransitionMatrix(
            root={
                ActivityLevel.SEDENTARY: {
                    Action.SEND: {
                        ActivityLevel.SEDENTARY: 2.0,
                        ActivityLevel.ACTIVE: -0.5,
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
        )


def test_time_of_day_count_mismatch_rejected():
    with pytest.raises(ValidationError, match="must match"):
        _make_config(steps_per_day=2)


def test_missing_transition_entry_rejected():
    bad_transition = TransitionMatrix(
        root={
            ActivityLevel.SEDENTARY: {
                Action.SEND: {ActivityLevel.SEDENTARY: 0.5, ActivityLevel.ACTIVE: 0.5},
            },
            ActivityLevel.ACTIVE: {
                Action.SEND: {ActivityLevel.SEDENTARY: 0.5, ActivityLevel.ACTIVE: 0.5},
                Action.DON_T_SEND: {
                    ActivityLevel.SEDENTARY: 0.5,
                    ActivityLevel.ACTIVE: 0.5,
                },
            },
        }
    )
    with pytest.raises(ValidationError, match="Missing transition entry"):
        _make_config(transition=bad_transition)
