import pytest

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.environment import Environment
from rl_health_interventions.state import StateView

_HERE = pytest.importorskip("pathlib").Path(__file__).resolve().parent.parent
ASSETS_TABLES = str(_HERE / "assets" / "tables")


def _mvp_config(steps_per_day=5, episode_days=90) -> MDPConfig:
    return MDPConfig(
        episode_days=episode_days,
        steps_per_day=steps_per_day,
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
        },
        transition_model={"type": "rule_based", "table_dir": ASSETS_TABLES},
    )


def _factored_config() -> MDPConfig:
    return MDPConfig(
        episode_days=1,
        steps_per_day=5,
        seed=42,
        initial_state={
            "step_bin": "inactive",
            "sleep": "rested",
            "day_of_week": "weekday",
            "burden": "low",
        },
        state={
            "factors": {
                "step_bin": {"dims": 3, "names": ["inactive", "moderate", "active"]},
                "sleep": {"dims": 2, "names": ["rested", "under_rested"]},
                "day_of_week": {"dims": 2, "names": ["weekday", "weekend"]},
                "burden": {"dims": 3, "names": ["low", "medium", "high"]},
            },
        },
        actions={
            "idle": {"action_penalty": 0},
            "movement_suggestion": {"action_penalty": 1},
            "goal_reminder": {"action_penalty": 1},
            "journal": {"action_penalty": 1},
        },
        reward={
            "factor": "step_bin",
            "values": {"inactive": 0.0, "moderate": 0.5, "active": 1.0},
            "action_penalty_multiplier": 0.05,
        },
        transition_model={"type": "rule_based", "table_dir": ASSETS_TABLES},
    )


def test_reset_returns_stateview(valid_config):
    env = Environment(valid_config, seed=42)
    state = env.reset()
    assert isinstance(state, StateView)
    assert state.activity_level == "sedentary"
    assert state.day == 0
    assert state.step_of_day == 0


def test_reset_includes_all_factors(valid_config):
    env = Environment(valid_config, seed=42)
    state = env.reset()
    assert hasattr(state, "activity_level")


def test_step_returns_tuple(valid_config):
    env = Environment(valid_config, seed=42)
    env.reset()
    next_state, reward, done = env.step("nudge")
    assert isinstance(next_state, StateView)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_episode_terminates_after_correct_length():
    env = Environment(_mvp_config(steps_per_day=5, episode_days=2), seed=42)
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
    env = Environment(_mvp_config(steps_per_day=2, episode_days=1), seed=42)
    env.reset()
    env.step("nudge")
    env.step("nudge")
    with pytest.raises(RuntimeError, match="Episode is done"):
        env.step("nudge")


def test_step_before_reset_raises():
    env = Environment(_mvp_config(), seed=42)
    with pytest.raises(RuntimeError, match="Call reset"):
        env.step("nudge")


def test_factored_env_step_returns_stateview(sprint1_config):
    env = Environment(sprint1_config, seed=42)
    env.reset()
    next_state, reward, done = env.step("idle")
    assert isinstance(next_state, StateView)
    assert hasattr(next_state, "step_bin")
    assert hasattr(next_state, "sleep")
    assert isinstance(reward, float)
    assert isinstance(done, bool)
