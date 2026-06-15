from rl_health_interventions.config.schemas import (
    Action,
    ActivityLevel,
    MDPConfig,
    TimeOfDay,
    TimeOfDayMask,
    TransitionMatrix,
)
from rl_health_interventions.environment import Environment

ALL_TIMES = [
    TimeOfDay.MORNING,
    TimeOfDay.MIDDAY,
    TimeOfDay.AFTERNOON,
    TimeOfDay.EVENING,
    TimeOfDay.NIGHT,
]


def _config(steps_per_day=5, episode_days=90) -> MDPConfig:
    return MDPConfig(
        activity_levels=[ActivityLevel.SEDENTARY, ActivityLevel.ACTIVE],
        actions=[Action.SEND, Action.DON_T_SEND],
        time_of_day=ALL_TIMES[:steps_per_day],
        steps_per_day=steps_per_day,
        episode_days=episode_days,
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
                t: {ActivityLevel.SEDENTARY: 0.0, ActivityLevel.ACTIVE: 0.0}
                for t in ALL_TIMES[:steps_per_day]
            }
        ),
    )


def test_reset_returns_initial_state(valid_config):
    env = Environment(valid_config, seed=42)
    state = env.reset()
    assert state.activity == ActivityLevel.SEDENTARY
    assert state.day == 0
    assert state.step_of_day == 0
    assert state.time_of_day == TimeOfDay.MORNING


def test_step_returns_tuple(valid_config):
    env = Environment(valid_config, seed=42)
    env.reset()
    next_state, reward, done = env.step(Action.SEND)
    assert isinstance(next_state.activity, ActivityLevel)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_episode_terminates_after_correct_length():
    env = Environment(_config(steps_per_day=5, episode_days=2), seed=42)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done = env.step(Action.SEND)
        steps += 1
        if steps > 100:
            break
    assert steps == 10
    assert done is True


def test_time_of_day_cycles_within_day():
    env = Environment(_config(steps_per_day=5, episode_days=1), seed=42)
    state = env.reset()
    expected_times = [
        TimeOfDay.MORNING,
        TimeOfDay.MIDDAY,
        TimeOfDay.AFTERNOON,
        TimeOfDay.EVENING,
        TimeOfDay.NIGHT,
    ]
    for i, expected in enumerate(expected_times):
        assert state.time_of_day == expected, f"step {i}"
        if i < 4:
            state, _, _ = env.step(Action.SEND)


def test_step_after_done_raises():
    import pytest

    env = Environment(_config(steps_per_day=2, episode_days=1), seed=42)
    env.reset()
    env.step(Action.SEND)
    env.step(Action.SEND)
    with pytest.raises(RuntimeError, match="Episode is done"):
        env.step(Action.SEND)


def test_step_before_reset_raises():
    import pytest

    env = Environment(_config(), seed=42)
    with pytest.raises(RuntimeError, match="Call reset"):
        env.step(Action.SEND)


def test_night_mask_suppresses_transitions():
    from rl_health_interventions.config.schemas import (
        ActivityLevel,
        Action,
        MDPConfig,
        TimeOfDay,
        TimeOfDayMask,
        TransitionMatrix,
    )

    # All masks = 1.0: transitions suppressed at every time slot
    config = MDPConfig(
        activity_levels=[ActivityLevel.SEDENTARY, ActivityLevel.ACTIVE],
        actions=[Action.SEND, Action.DON_T_SEND],
        time_of_day=[
            TimeOfDay.MORNING,
            TimeOfDay.MIDDAY,
            TimeOfDay.AFTERNOON,
            TimeOfDay.EVENING,
            TimeOfDay.NIGHT,
        ],
        steps_per_day=5,
        episode_days=10,
        transition=TransitionMatrix(
            root={
                ActivityLevel.SEDENTARY: {
                    Action.SEND: {
                        ActivityLevel.SEDENTARY: 0.0,
                        ActivityLevel.ACTIVE: 1.0,
                    },
                    Action.DON_T_SEND: {
                        ActivityLevel.SEDENTARY: 0.0,
                        ActivityLevel.ACTIVE: 1.0,
                    },
                },
                ActivityLevel.ACTIVE: {
                    Action.SEND: {
                        ActivityLevel.SEDENTARY: 1.0,
                        ActivityLevel.ACTIVE: 0.0,
                    },
                    Action.DON_T_SEND: {
                        ActivityLevel.SEDENTARY: 1.0,
                        ActivityLevel.ACTIVE: 0.0,
                    },
                },
            }
        ),
        masks=TimeOfDayMask(
            root={
                t: {ActivityLevel.SEDENTARY: 1.0, ActivityLevel.ACTIVE: 1.0}
                for t in TimeOfDay
            }
        ),
    )
    env = Environment(config, seed=42)
    state = env.reset()
    base_state = state.activity
    # Transition matrix wants to flip state every step (P=1.0 for opposite),
    # but masks suppress all transitions. State should stay the same.
    for _ in range(49):
        state, _, _ = env.step(Action.SEND)
        assert state.activity == base_state
