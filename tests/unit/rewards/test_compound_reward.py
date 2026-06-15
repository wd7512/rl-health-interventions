from rl_health_interventions.config.schemas import (
    ActivityLevel,
    Action,
    MDPConfig,
    TransitionMatrix,
    TimeOfDayMask,
    TimeOfDay,
)
from rl_health_interventions.rewards.compound import CompoundReward


def _config() -> MDPConfig:
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


def test_active_state_reward():
    r = CompoundReward(_config())
    reward, done = r.reward(ActivityLevel.ACTIVE, Action.SEND)
    assert reward == 1.0
    assert done is False


def test_sedentary_state_reward():
    r = CompoundReward(_config())
    reward, done = r.reward(ActivityLevel.SEDENTARY, Action.SEND)
    assert reward == 0.0
    assert done is False


def test_reward_never_terminates():
    r = CompoundReward(_config())
    for _ in range(10):
        _, done = r.reward(ActivityLevel.SEDENTARY, Action.SEND)
        assert done is False
