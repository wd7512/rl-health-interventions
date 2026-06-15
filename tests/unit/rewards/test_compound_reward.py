from rl_health_interventions.config.schemas import (
    ActivityLevel,
    Action,
)
from rl_health_interventions.rewards.compound import CompoundReward


def test_active_state_reward(minimal_config):
    r = CompoundReward(minimal_config)
    reward, done = r.reward(ActivityLevel.ACTIVE, Action.SEND)
    assert reward == 1.0
    assert done is False


def test_sedentary_state_reward(minimal_config):
    r = CompoundReward(minimal_config)
    reward, done = r.reward(ActivityLevel.SEDENTARY, Action.SEND)
    assert reward == 0.0
    assert done is False


def test_reward_never_terminates(minimal_config):
    r = CompoundReward(minimal_config)
    for _ in range(10):
        _, done = r.reward(ActivityLevel.SEDENTARY, Action.SEND)
        assert done is False
