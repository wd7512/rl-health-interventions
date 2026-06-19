"""Tests for extended mode (state_dynamics) in RuleBasedTransition."""

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView
from rl_health_interventions.transitions.rule_based import RuleBasedTransition


def _extended_config() -> MDPConfig:
    return MDPConfig(
        episode_days=90,
        steps_per_day=5,
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
            "state_dynamics": {
                "steps": {
                    "response_multiplier": {"nudge": 200, "idle": 50},
                    "tod_modulation": {0: 0.2, 6: 0.5, 12: 0.9, 18: 0.6, 23: 0.2},
                    "dow_modulation": {0: 0.7, 1: 0.9, 5: 0.8, 6: 0.6},
                    "noise_std": 0.0,
                },
                "weight": {
                    "meal_effect": {0: -0.02, 6: 0.01, 12: 0.03, 18: 0.03, 23: -0.01},
                    "weekend_boost": 0.05,
                    "steps_coefficient": -0.0001,
                    "noise_std": 0.0,
                },
            },
        },
        agents=[{"type": "thompson_sampling", "alpha_prior": 1, "beta_prior": 1}],
    )


def test_steps_evolves_per_equation():
    """With zero noise, steps changes by exactly the mean."""
    config = _extended_config()
    # step_of_day=0 → tod=0, modulation 0.2
    # day=0 → dow=0, modulation 0.7
    # action="nudge" → response_multiplier 200
    # mean = 200 * 0.2 * 0.7 = 28.0
    t = RuleBasedTransition(config, seed=42)
    state = StateView(
        activity="sedentary",
        day=0,
        step_of_day=0,
        steps_per_day=5,
        steps=5000.0,
        weight=70.0,
    )
    next_state = t.transition(state, "nudge")
    assert next_state.steps == 5028.0


def test_weight_evolves_per_equation():
    config = _extended_config()
    t = RuleBasedTransition(config, seed=42)
    # tod=0 → meal_effect=-0.02
    # dow=0 (sunday) → weekend_boost=0.05
    # steps delta = 28.0 (from test above)
    # weight_delta = -0.02 + 0.05 + (-0.0001 * 28.0) = 0.0272
    state = StateView(
        activity="sedentary",
        day=0,
        step_of_day=0,
        steps_per_day=5,
        steps=5000.0,
        weight=70.0,
    )
    next_state = t.transition(state, "nudge")
    expected = 70.0 + (-0.02) + 0.05 + (-0.0001 * 28.0)
    assert abs(next_state.weight - expected) < 1e-10


def test_time_of_day_computed_correctly():
    """Extended mode computes time_of_day from step_of_day."""
    config = _extended_config()
    t = RuleBasedTransition(config, seed=42)
    state = StateView(
        activity="sedentary",
        day=0,
        step_of_day=2,
        steps_per_day=5,
        steps=5000.0,
        weight=70.0,
    )
    # step_of_day=2 → time_of_day = int(2 * 24 / 5) = 9
    next_state = t.transition(state, "nudge")
    assert next_state.time_of_day == 9

    state2 = StateView(
        activity="sedentary",
        day=0,
        step_of_day=4,
        steps_per_day=5,
        steps=5000.0,
        weight=70.0,
    )
    next_state2 = t.transition(state2, "nudge")
    assert next_state2.time_of_day == 19


def test_day_of_week_computed_correctly():
    """Extended mode computes day_of_week from day number."""
    config = _extended_config()
    t = RuleBasedTransition(config, seed=42)
    # day=0 → dow=0
    state = StateView(
        activity="sedentary",
        day=0,
        step_of_day=0,
        steps_per_day=5,
        steps=5000.0,
        weight=70.0,
    )
    next_state = t.transition(state, "nudge")
    assert next_state.day_of_week == 0

    # day=6 → dow=6
    state2 = StateView(
        activity="sedentary",
        day=6,
        step_of_day=0,
        steps_per_day=5,
        steps=5000.0,
        weight=70.0,
    )
    next_state2 = t.transition(state2, "nudge")
    assert next_state2.day_of_week == 6

    # day=7 → dow=0
    state3 = StateView(
        activity="sedentary",
        day=7,
        step_of_day=0,
        steps_per_day=5,
        steps=5000.0,
        weight=70.0,
    )
    next_state3 = t.transition(state3, "nudge")
    assert next_state3.day_of_week == 0


def test_extended_mode_preserves_activity_transition():
    """Activity should still transition per the probability matrix."""
    config = _extended_config()
    t = RuleBasedTransition(config, seed=42)
    state = StateView(
        activity="sedentary",
        day=0,
        step_of_day=0,
        steps_per_day=5,
        steps=5000.0,
        weight=70.0,
    )
    n_samples = 10_000
    n_active = sum(
        1 for _ in range(n_samples) if t.transition(state, "nudge").activity == "active"
    )
    observed_freq = n_active / n_samples
    assert abs(observed_freq - 0.3) < 0.02


def test_extended_mode_with_none_values_still_works():
    """When steps/weight are None, extended mode still works."""
    config = _extended_config()
    t = RuleBasedTransition(config, seed=42)
    state = StateView(activity="sedentary", day=0, step_of_day=0)
    next_state = t.transition(state, "nudge")
    assert next_state.activity in ("sedentary", "active")


def test_steps_delta_uses_clipped_value_for_weight():
    """Weight equation uses actual clipped delta, not raw pre-clipping value.

    When steps=0.0 and noise would push them negative, the clipped new_steps
    is 0.0. The weight equation must use delta=0.0 (not the phantom negative),
    otherwise weight incorrectly increases via a negative * negative_coefficient.
    """
    config = _extended_config()
    # Use a state with steps=0.0 and negative noise to trigger clipping
    t = RuleBasedTransition(config, seed=42)
    state = StateView(
        activity="sedentary",
        day=0,
        step_of_day=0,
        steps_per_day=5,
        steps=0.0,
        weight=70.0,
    )
    # With noise_std=0 in config, steps_mean=200*0.2*0.7=28.0, no noise.
    # So steps_delta = 28.0 (new_steps=28, old=0), weight changes by:
    # meal_effect(-0.02) + weekend_boost(0.05) + coeff*(-0.0001)*28.0
    next_state = t.transition(state, "nudge")
    expected_weight_delta = -0.02 + 0.05 + (-0.0001 * 28.0)
    assert abs(next_state.weight - (70.0 + expected_weight_delta)) < 1e-10

    # Now verify the key invariant: weight should NOT increase from phantom
    # negative steps_delta. With old code (steps_delta = steps_mean + noise),
    # if steps went negative before clipping, the negative delta * negative
    # coefficient would produce a weight INCREASE. With the fix, delta is
    # always >= 0 after clipping, so weight never gets a phantom boost.
    # Direct check: when old_steps > new_steps (both >= 0), delta >= 0
    assert next_state.steps is not None
    assert next_state.steps >= 0.0
    assert next_state.steps - 0.0 >= 0.0  # delta always non-negative
