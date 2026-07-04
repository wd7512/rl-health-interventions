from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.rewards.expression import ExpressionReward

_BASE = {
    "episode_days": 1,
    "steps_per_day": 5,
    "seed": 42,
    "state": {"variables": {"activity_level": {"names": ["sedentary", "active"]}}},
    "initial_state": {"activity_level": "sedentary"},
    "actions": ["nudge", "idle"],
    "reward": {
        "variables": {
            "value": {
                "source": "state.activity_level",
                "mapping": {"sedentary": 0.0, "active": 1.0},
            }
        },
        "formula": "value",
    },
    "transition_model": {
        "type": "rule_based",
        "transition_probabilities": {
            "sedentary": {
                "nudge": {"active": 0.3, "sedentary": 0.7},
                "idle": {"active": 0.1, "sedentary": 0.9},
            },
            "active": {
                "nudge": {"active": 0.5, "sedentary": 0.5},
                "idle": {"active": 0.6, "sedentary": 0.4},
            },
        },
    },
}


def _config(reward_multiplier=None) -> MDPConfig:
    kw = dict(_BASE)
    if reward_multiplier is not None:
        kw["reward_multiplier_by_step"] = reward_multiplier
    return MDPConfig(**kw)


def test_active_state_reward(active_state):
    r = ExpressionReward(_config())
    reward, done = r.reward(active_state, "nudge", step_idx=0)
    assert reward == 1.0
    assert done is False


def test_sedentary_state_reward(sedentary_state):
    r = ExpressionReward(_config())
    reward, done = r.reward(sedentary_state, "nudge", step_idx=0)
    assert reward == 0.0
    assert done is False


def test_reward_never_terminates(sedentary_state):
    r = ExpressionReward(_config())
    for _ in range(10):
        _, done = r.reward(sedentary_state, "nudge", step_idx=0)
        assert done is False


def test_reward_multiplier(active_state):
    r = ExpressionReward(_config(reward_multiplier=[1.0, 0.5, 0.0, 0.0, 0.0]))
    assert r.reward(active_state, "nudge", step_idx=0) == (1.0, False)
    assert r.reward(active_state, "nudge", step_idx=1) == (0.5, False)
    assert r.reward(active_state, "nudge", step_idx=2) == (0.0, False)


def test_reward_with_string_state():
    r = ExpressionReward(_config())
    reward, done = r.reward("active", "nudge", step_idx=0)
    assert reward == 1.0
    assert done is False
