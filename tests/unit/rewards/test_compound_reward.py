import pytest

from rl_health_interventions.rewards.compound import CompoundReward
from rl_health_interventions.state import StateView


def _sv(activity: str, **kwargs) -> StateView:
    """Helper to build a StateView with defaults: day=0, step_of_day=0, steps_per_day=1."""
    return StateView(
        activity=activity,
        day=kwargs.get("day", 0),
        step_of_day=kwargs.get("step_of_day", 0),
        steps_per_day=kwargs.get("steps_per_day", 1),
    )


def test_active_state_reward(minimal_config):
    r = CompoundReward(minimal_config)
    reward, done = r.reward(_sv("active"), "nudge", step_idx=0)
    assert reward == 1.0
    assert done is False


def test_sedentary_state_reward(minimal_config):
    r = CompoundReward(minimal_config)
    reward, done = r.reward(_sv("sedentary"), "nudge", step_idx=0)
    assert reward == 0.0
    assert done is False


def test_reward_never_terminates(minimal_config):
    r = CompoundReward(minimal_config)
    for _ in range(10):
        _, done = r.reward(_sv("sedentary"), "nudge", step_idx=0)
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
    assert r.reward(_sv("active", steps_per_day=3), "nudge", step_idx=0) == (1.0, False)
    assert r.reward(_sv("active", steps_per_day=3), "nudge", step_idx=1) == (0.5, False)
    assert r.reward(_sv("active", steps_per_day=3), "nudge", step_idx=2) == (0.0, False)


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


# --- Multi-timescale reward tests ---


def test_simple_mode_mvp_identical(minimal_config):
    """Config without reward_weights reproduces simple per-step reward."""
    r = CompoundReward(minimal_config)
    rew1, done1 = r.reward(_sv("active"), "nudge", step_idx=0)
    rew2, done2 = r.reward(_sv("sedentary"), "nudge", step_idx=0)
    assert rew1 == 1.0
    assert rew2 == 0.0
    assert done1 is False
    assert done2 is False


def test_flat_bonus_at_interval():
    """Option A: flat delayed_reward_value fires at interval boundary."""
    from rl_health_interventions.config.schemas import MDPConfig, RewardWeightsConfig

    config = MDPConfig(
        episode_days=10,
        steps_per_day=1,
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
        reward_weights=RewardWeightsConfig(
            mode="multi_timescale",
            delayed_reward_interval=3,
            delayed_reward_value=10.0,
        ),
    )
    r = CompoundReward(config)
    # global_step=3 is the boundary (3 % 3 == 0)
    rew, done = r.reward(
        _sv("active", day=0, step_of_day=3, steps_per_day=1), "nudge", step_idx=0
    )
    assert rew == 1.0 + 10.0
    assert done is False


def test_no_bonus_off_interval():
    """Option A: no bonus when global_step is not at interval boundary."""
    from rl_health_interventions.config.schemas import MDPConfig, RewardWeightsConfig

    config = MDPConfig(
        episode_days=10,
        steps_per_day=1,
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
        reward_weights=RewardWeightsConfig(
            mode="multi_timescale",
            delayed_reward_interval=3,
            delayed_reward_value=10.0,
        ),
    )
    r = CompoundReward(config)
    # global_step=2 is not a boundary (2 % 3 != 0)
    rew, done = r.reward(
        _sv("active", day=0, step_of_day=2, steps_per_day=1), "nudge", step_idx=0
    )
    assert rew == 1.0  # no bonus
    assert done is False


def test_scaled_bonus_depends_on_activity():
    """Option B: more active steps → higher scaled bonus at interval boundary."""
    from rl_health_interventions.config.schemas import MDPConfig, RewardWeightsConfig

    config = MDPConfig(
        episode_days=10,
        steps_per_day=1,
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
        reward_weights=RewardWeightsConfig(
            mode="multi_timescale",
            delayed_reward_interval=3,
            delayed_reward_value=10.0,
            delayed_reward_scale=6.0,
            delayed_reward_threshold=0.0,
        ),
    )
    r = CompoundReward(config)
    # Simulate 2 active steps out of 3 in the interval (boundary step is also counted)
    r.reward(_sv("active", day=0, step_of_day=1, steps_per_day=1), "nudge", step_idx=0)
    r.reward(_sv("active", day=0, step_of_day=2, steps_per_day=1), "nudge", step_idx=0)
    # Boundary step: global_step=3, all 3 active, rate=3/3=1.0, bonus=6.0*1.0=6.0
    rew, done = r.reward(
        _sv("active", day=0, step_of_day=3, steps_per_day=1), "nudge", step_idx=0
    )
    assert rew == 1.0 + 6.0
    assert done is False


def test_scaled_bonus_partial_active():
    """Option B: partial activity gives proportional bonus."""
    from rl_health_interventions.config.schemas import MDPConfig, RewardWeightsConfig

    config = MDPConfig(
        episode_days=10,
        steps_per_day=1,
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
        reward_weights=RewardWeightsConfig(
            mode="multi_timescale",
            delayed_reward_interval=3,
            delayed_reward_value=10.0,
            delayed_reward_scale=6.0,
            delayed_reward_threshold=0.0,
        ),
    )
    r = CompoundReward(config)
    # 2 active out of 3 in the interval (boundary included)
    r.reward(_sv("active", day=0, step_of_day=1, steps_per_day=1), "nudge", step_idx=0)
    r.reward(
        _sv("sedentary", day=0, step_of_day=2, steps_per_day=1), "nudge", step_idx=0
    )
    # Boundary: global_step=3, 2 active / 3 total = 0.666..., bonus = 6.0 * 2/3 = 4.0
    rew, done = r.reward(
        _sv("active", day=0, step_of_day=3, steps_per_day=1), "nudge", step_idx=0
    )
    assert rew == pytest.approx(1.0 + 4.0)
    assert done is False


def test_scaled_bonus_zero_below_threshold():
    """Option B: rate below threshold → zero bonus at interval boundary."""
    from rl_health_interventions.config.schemas import MDPConfig, RewardWeightsConfig

    config = MDPConfig(
        episode_days=10,
        steps_per_day=1,
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
        reward_weights=RewardWeightsConfig(
            mode="multi_timescale",
            delayed_reward_interval=3,
            delayed_reward_value=10.0,
            delayed_reward_scale=6.0,
            delayed_reward_threshold=0.5,
        ),
    )
    r = CompoundReward(config)
    # Simulate 1 active step out of 3 → rate=1/3 < 0.5 → bonus=0
    r.reward(_sv("active", day=0, step_of_day=1, steps_per_day=1), "nudge", step_idx=0)
    r.reward(
        _sv("sedentary", day=0, step_of_day=2, steps_per_day=1), "nudge", step_idx=0
    )
    # Boundary step: global_step=3, rate=1/3 < 0.5, bonus=0
    rew, done = r.reward(
        _sv("sedentary", day=0, step_of_day=3, steps_per_day=1), "nudge", step_idx=0
    )
    assert rew == 0.0  # sedentary base + zero bonus
    assert done is False


def test_reset_clears_active_count():
    """reset() clears the active count between episodes."""
    from rl_health_interventions.config.schemas import MDPConfig, RewardWeightsConfig

    config = MDPConfig(
        episode_days=10,
        steps_per_day=1,
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
        reward_weights=RewardWeightsConfig(
            mode="multi_timescale",
            delayed_reward_interval=3,
            delayed_reward_value=10.0,
            delayed_reward_scale=6.0,
            delayed_reward_threshold=0.0,
        ),
    )
    r = CompoundReward(config)
    # First episode: 2 active steps
    r.reward(_sv("active", step_of_day=1), "nudge", step_idx=0)
    r.reward(_sv("active", step_of_day=2), "nudge", step_idx=0)
    r.reset()
    # Second episode: 0 active steps before boundary → rate=0/3=0
    rew, _ = r.reward(_sv("sedentary", step_of_day=3), "nudge", step_idx=0)
    assert rew == 0.0  # 0 base + 0 bonus (rate 0/3)
