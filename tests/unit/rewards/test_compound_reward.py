import pytest

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.rewards.compound import CompoundReward
from rl_health_interventions.state import StateView


def test_active_state_reward(minimal_config):
    r = CompoundReward(minimal_config)
    state = StateView({"activity_level": "active"}, day=0, step_of_day=0)
    reward, done = r.reward(state, "nudge", step_idx=0)
    assert reward == 1.0
    assert done is False


def test_sedentary_state_reward(minimal_config):
    r = CompoundReward(minimal_config)
    state = StateView({"activity_level": "sedentary"}, day=0, step_of_day=0)
    reward, done = r.reward(state, "nudge", step_idx=0)
    assert reward == 0.0
    assert done is False


def test_reward_never_terminates(minimal_config):
    r = CompoundReward(minimal_config)
    state = StateView({"activity_level": "sedentary"}, day=0, step_of_day=0)
    for _ in range(10):
        _, done = r.reward(state, "nudge", step_idx=0)
        assert done is False


def test_per_step_reward_uses_step_index():
    config = MDPConfig(
        episode_days=1,
        steps_per_day=3,
        seed=42,
        initial_state={"activity_level": "sedentary"},
        state={
            "factors": {
                "activity_level": {"dims": 2, "names": ["sedentary", "active"]},
            },
        },
        actions=["nudge", "idle"],
        reward={
            "factor": "activity_level",
            "values": {"sedentary": 0.0, "active": 1.0},
            "per_step_multiplier": [1.0, 0.5, 0.0],
        },
        transition_model={"type": "rule_based"},
    )
    r = CompoundReward(config)
    state = StateView({"activity_level": "active"}, day=0, step_of_day=0)
    assert r.reward(state, "nudge", step_idx=0) == (1.0, False)
    assert r.reward(state, "nudge", step_idx=1) == (0.5, False)
    assert r.reward(state, "nudge", step_idx=2) == (0.0, False)


def test_action_penalty():
    config = MDPConfig(
        episode_days=1,
        steps_per_day=1,
        seed=42,
        initial_state={"activity_level": "sedentary"},
        state={
            "factors": {
                "activity_level": {"dims": 2, "names": ["sedentary", "active"]},
            },
        },
        actions={"nudge": {"action_penalty": 1.0}, "idle": {"action_penalty": 0.0}},
        reward={
            "factor": "activity_level",
            "values": {"sedentary": 0.0, "active": 1.0},
            "action_penalty_multiplier": 0.5,
        },
        transition_model={"type": "rule_based"},
    )
    r = CompoundReward(config)
    state = StateView({"activity_level": "active"}, day=0, step_of_day=0)
    # active=1.0 - nudge_penalty=1.0 * mult=0.5 = 1.0 - 0.5 = 0.5
    reward, _ = r.reward(state, "nudge", step_idx=0)
    assert reward == 0.5
    # idle: active=1.0 - idle_penalty=0.0 * mult=0.5 = 1.0
    reward, _ = r.reward(state, "idle", step_idx=0)
    assert reward == 1.0


def test_schema_ref_config_raises_not_implemented():
    from rl_health_interventions.config.schemas import MDPConfig

    config = MDPConfig(
        episode_days=1,
        steps_per_day=1,
        seed=42,
        initial_state={"activity_level": "sedentary"},
        state={"schema": "heartsteps"},
        actions=["nudge", "idle"],
        reward={
            "factor": "activity_level",
            "values": {"sedentary": 0.0, "active": 1.0},
        },
        transition_model={"type": "learned"},
    )
    with pytest.raises(NotImplementedError, match="Schema-ref"):
        CompoundReward(config)
