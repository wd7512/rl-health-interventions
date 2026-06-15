import pytest

from rl_health_interventions.config.schemas import (
    ActivityLevel,
    Action,
    MDPConfig,
    TimeOfDay,
    TimeOfDayMask,
    TransitionMatrix,
)

ALL_TIMES = [
    TimeOfDay.MORNING,
    TimeOfDay.MIDDAY,
    TimeOfDay.AFTERNOON,
    TimeOfDay.EVENING,
    TimeOfDay.NIGHT,
]


@pytest.fixture
def valid_config() -> MDPConfig:
    return MDPConfig(
        activity_levels=[ActivityLevel.SEDENTARY, ActivityLevel.ACTIVE],
        actions=[Action.SEND, Action.DON_T_SEND],
        time_of_day=ALL_TIMES,
        steps_per_day=5,
        episode_days=90,
        transition=TransitionMatrix(
            root={
                ActivityLevel.SEDENTARY: {
                    Action.SEND: {
                        ActivityLevel.SEDENTARY: 0.7,
                        ActivityLevel.ACTIVE: 0.3,
                    },
                    Action.DON_T_SEND: {
                        ActivityLevel.SEDENTARY: 0.9,
                        ActivityLevel.ACTIVE: 0.1,
                    },
                },
                ActivityLevel.ACTIVE: {
                    Action.SEND: {
                        ActivityLevel.SEDENTARY: 0.2,
                        ActivityLevel.ACTIVE: 0.8,
                    },
                    Action.DON_T_SEND: {
                        ActivityLevel.SEDENTARY: 0.4,
                        ActivityLevel.ACTIVE: 0.6,
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
                TimeOfDay.MIDDAY: {
                    ActivityLevel.SEDENTARY: 0.0,
                    ActivityLevel.ACTIVE: 0.0,
                },
                TimeOfDay.AFTERNOON: {
                    ActivityLevel.SEDENTARY: 0.0,
                    ActivityLevel.ACTIVE: 0.0,
                },
                TimeOfDay.EVENING: {
                    ActivityLevel.SEDENTARY: 0.0,
                    ActivityLevel.ACTIVE: 0.0,
                },
                TimeOfDay.NIGHT: {
                    ActivityLevel.SEDENTARY: 1.0,
                    ActivityLevel.ACTIVE: 1.0,
                },
            }
        ),
        seed=42,
    )


@pytest.fixture
def minimal_config() -> MDPConfig:
    return MDPConfig(
        activity_levels=[ActivityLevel.SEDENTARY, ActivityLevel.ACTIVE],
        actions=[Action.SEND, Action.DON_T_SEND],
        time_of_day=[TimeOfDay.MORNING],
        steps_per_day=1,
        episode_days=1,
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
