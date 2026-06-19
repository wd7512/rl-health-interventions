from rl_health_interventions.rewards.compound import CompoundReward
from rl_health_interventions.state import StateView


def test_active_state_reward(minimal_config):
    r = CompoundReward(minimal_config)
    state = StateView("active", 0, 0)
    reward, done = r.reward(state, "nudge", step_idx=0)
    assert reward == 1.0
    assert done is False


def test_sedentary_state_reward(minimal_config):
    r = CompoundReward(minimal_config)
    state = StateView("sedentary", 0, 0)
    reward, done = r.reward(state, "nudge", step_idx=0)
    assert reward == 0.0
    assert done is False


def test_reward_never_terminates(minimal_config):
    r = CompoundReward(minimal_config)
    state = StateView("sedentary", 0, 0)
    for _ in range(10):
        _, done = r.reward(state, "nudge", step_idx=0)
        assert done is False


def test_per_step_reward_uses_step_index():
    from rl_health_interventions.config.schemas import MDPConfig

    config = MDPConfig(
        episode_days=1,
        steps_per_day=3,
        seed=42,
        initial_state="sedentary",
        states={"sedentary": {"reward": 0.0}, "active": {"reward": 1.0}},
        actions=["nudge", "idle"],
        transition_model={
            "type": "rule_based",
            "transition_probabilities": {
                "sedentary": {
                    "nudge": {"active": 0.5, "sedentary": 0.5},
                    "idle": {"active": 0.5, "sedentary": 0.5},
                },
                "active": {
                    "nudge": {"active": 0.5, "sedentary": 0.5},
                    "idle": {"active": 0.5, "sedentary": 0.5},
                },
            },
        },
        reward_multiplier_by_step=[1.0, 0.5, 0.0],
    )
    r = CompoundReward(config)
    assert r.reward(StateView("active", 0, 0), "nudge", step_idx=0) == (1.0, False)
    assert r.reward(StateView("active", 0, 0), "nudge", step_idx=1) == (0.5, False)
    assert r.reward(StateView("active", 0, 0), "nudge", step_idx=2) == (0.0, False)


def test_schema_ref_config_raises_not_implemented():
    import pytest
    from rl_health_interventions.config.schemas import MDPConfig

    config = MDPConfig(
        episode_days=1,
        steps_per_day=1,
        seed=42,
        initial_state="sedentary",
        states={"schema": "heartsteps"},
        actions={"schema": "heartsteps"},
        transition_model={"type": "learned"},
    )
    with pytest.raises(NotImplementedError, match="Schema-ref"):
        CompoundReward(config)
