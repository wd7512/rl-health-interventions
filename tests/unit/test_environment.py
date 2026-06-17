from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.environment import Environment


def _config(steps_per_day=5, episode_days=90) -> MDPConfig:
    return MDPConfig(
        episode_days=episode_days,
        steps_per_day=steps_per_day,
        seed=42,
        initial_state="sedentary",
        states={"sedentary": {"reward": 0.0}, "active": {"reward": 1.0}},
        actions=["nudge", "idle"],
        transition_model={
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
    )


def test_reset_returns_initial_state(valid_config):
    env = Environment(valid_config, seed=42)
    state = env.reset()
    assert state.activity == "sedentary"
    assert state.day == 0
    assert state.step_of_day == 0


def test_step_returns_tuple(valid_config):
    env = Environment(valid_config, seed=42)
    env.reset()
    next_state, reward, done = env.step("nudge")
    assert isinstance(next_state.activity, str)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_episode_terminates_after_correct_length():
    env = Environment(_config(steps_per_day=5, episode_days=2), seed=42)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done = env.step("nudge")
        steps += 1
        if steps > 100:
            break
    assert steps == 10
    assert done is True


def test_step_after_done_raises():
    import pytest

    env = Environment(_config(steps_per_day=2, episode_days=1), seed=42)
    env.reset()
    env.step("nudge")
    env.step("nudge")
    with pytest.raises(RuntimeError, match="Episode is done"):
        env.step("nudge")


def test_step_before_reset_raises():
    import pytest

    env = Environment(_config(), seed=42)
    with pytest.raises(RuntimeError, match="Call reset"):
        env.step("nudge")


def test_reward_multiplier_affects_reward():
    from rl_health_interventions.config.schemas import MDPConfig

    # Deterministic transitions: always go to "active"
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
                    "nudge": {"active": 1.0, "sedentary": 0.0},
                    "idle": {"active": 1.0, "sedentary": 0.0},
                },
                "active": {
                    "nudge": {"active": 1.0, "sedentary": 0.0},
                    "idle": {"active": 1.0, "sedentary": 0.0},
                },
            },
        },
        reward_multiplier_by_step=[1.0, 1.0, 0.0],
    )
    env = Environment(config, seed=42)
    env.reset()
    _, reward_0, _ = env.step("nudge")  # step_idx=0, always active → 1.0 * 1.0
    _, reward_1, _ = env.step("nudge")  # step_idx=1, always active → 1.0 * 1.0
    _, reward_2, _ = env.step("nudge")  # step_idx=2, always active → 1.0 * 0.0
    assert reward_0 == 1.0
    assert reward_1 == 1.0
    assert reward_2 == 0.0
