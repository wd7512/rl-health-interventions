import pytest
from pydantic import ValidationError

from rl_health_interventions.config.schemas import (
    MDPConfig, ActivityLevel, Action, TimeOfDay,
    TransitionMatrix, TimeOfDayMask,
)


VALID_TRANSITION = TransitionMatrix(root={
    ActivityLevel.SEDENTARY: {
        Action.SEND: {ActivityLevel.SEDENTARY: 0.5, ActivityLevel.ACTIVE: 0.5},
        Action.DON_T_SEND: {ActivityLevel.SEDENTARY: 0.5, ActivityLevel.ACTIVE: 0.5},
    },
    ActivityLevel.ACTIVE: {
        Action.SEND: {ActivityLevel.SEDENTARY: 0.5, ActivityLevel.ACTIVE: 0.5},
        Action.DON_T_SEND: {ActivityLevel.SEDENTARY: 0.5, ActivityLevel.ACTIVE: 0.5},
    },
})

VALID_MASKS = TimeOfDayMask(root={
    TimeOfDay.MORNING: {ActivityLevel.SEDENTARY: 0.0, ActivityLevel.ACTIVE: 0.0},
})


def test_negative_probability_rejected():
    """Negative probability caught at TransitionMatrix level."""
    with pytest.raises(ValidationError, match="cannot be negative"):
        TransitionMatrix(root={
            ActivityLevel.SEDENTARY: {
                Action.SEND: {ActivityLevel.SEDENTARY: 2.0, ActivityLevel.ACTIVE: -0.5},
                Action.DON_T_SEND: {ActivityLevel.SEDENTARY: 0.5, ActivityLevel.ACTIVE: 0.5},
            },
            ActivityLevel.ACTIVE: {
                Action.SEND: {ActivityLevel.SEDENTARY: 0.5, ActivityLevel.ACTIVE: 0.5},
                Action.DON_T_SEND: {ActivityLevel.SEDENTARY: 0.5, ActivityLevel.ACTIVE: 0.5},
            },
        })


def test_time_of_day_count_mismatch_rejected():
    with pytest.raises(ValidationError, match="must match"):
        MDPConfig(
            activity_levels=[ActivityLevel.SEDENTARY, ActivityLevel.ACTIVE],
            actions=[Action.SEND, Action.DON_T_SEND],
            time_of_day=[TimeOfDay.MORNING],
            steps_per_day=2,
            transition=VALID_TRANSITION,
            masks=VALID_MASKS,
        )


def test_missing_transition_entry_rejected():
    bad_transition = TransitionMatrix(root={
        ActivityLevel.SEDENTARY: {
            Action.SEND: {ActivityLevel.SEDENTARY: 0.5, ActivityLevel.ACTIVE: 0.5},
        },
        ActivityLevel.ACTIVE: {
            Action.SEND: {ActivityLevel.SEDENTARY: 0.5, ActivityLevel.ACTIVE: 0.5},
            Action.DON_T_SEND: {ActivityLevel.SEDENTARY: 0.5, ActivityLevel.ACTIVE: 0.5},
        },
    })
    with pytest.raises(ValidationError, match="Missing transition entry"):
        MDPConfig(
            activity_levels=[ActivityLevel.SEDENTARY, ActivityLevel.ACTIVE],
            actions=[Action.SEND, Action.DON_T_SEND],
            time_of_day=[TimeOfDay.MORNING],
            steps_per_day=1,
            transition=bad_transition,
            masks=VALID_MASKS,
        )
