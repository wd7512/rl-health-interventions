from rl_health_interventions.state import StateView
from rl_health_interventions.transitions.rule_based import RuleBasedTransition


def test_transition_returns_valid_state(valid_config):
    t = RuleBasedTransition(valid_config, seed=42)
    state = StateView({"activity_level": "sedentary"}, day=0, step_of_day=0)
    next_state = t.transition(state, "nudge")
    assert hasattr(next_state, "activity_level")
    assert next_state.activity_level in ("sedentary", "active")


def test_transition_probabilities_match_config_over_many_samples(valid_config):
    t = RuleBasedTransition(valid_config, seed=42)
    state = StateView({"activity_level": "sedentary"}, day=0, step_of_day=0)
    n_samples = 10_000
    n_active = sum(
        1
        for _ in range(n_samples)
        if t.transition(state, "nudge").activity_level == "active"
    )
    observed_freq = n_active / n_samples
    assert abs(observed_freq - 0.3) < 0.02


def test_transition_returns_stateview(valid_config):
    t = RuleBasedTransition(valid_config, seed=42)
    state = StateView({"activity_level": "sedentary"}, day=0, step_of_day=0)
    result = t.transition(state, "nudge")
    assert isinstance(result, StateView)


def test_factored_format_step0_updates_sleep_and_step_bin(sprint1_config):
    t = RuleBasedTransition(sprint1_config, seed=42)
    state = StateView(
        {
            "step_bin": "inactive",
            "sleep": "rested",
            "day_of_week": "weekday",
            "burden": "low",
        },
        day=0,
        step_of_day=0,
    )
    result = t.transition(state, "idle")
    assert result.step_bin in ("inactive", "moderate", "active")
    assert result.sleep in ("rested", "under_rested")
    assert result.day_of_week == "weekday"
    assert result.burden == "low"


def test_factored_format_step1_updates_only_step_bin(sprint1_config):
    t = RuleBasedTransition(sprint1_config, seed=42)
    state = StateView(
        {
            "step_bin": "inactive",
            "sleep": "rested",
            "day_of_week": "weekday",
            "burden": "low",
        },
        day=0,
        step_of_day=1,
    )
    result = t.transition(state, "movement_suggestion")
    # step_of_day=1 uses within_day_1 table, output is step_bin
    assert result.step_bin in ("inactive", "moderate", "active")
    assert result.sleep == "rested"  # unchanged


def test_factored_format_step2_preserves_sleep_and_other_factors(sprint1_config):
    t = RuleBasedTransition(sprint1_config, seed=42)
    state = StateView(
        {
            "step_bin": "moderate",
            "sleep": "under_rested",
            "day_of_week": "weekday",
            "burden": "low",
        },
        day=0,
        step_of_day=2,
    )
    result = t.transition(state, "journal")
    assert result.sleep == "under_rested"
    assert result.day_of_week == "weekday"
    assert result.burden == "low"
    assert result.step_bin in ("inactive", "moderate", "active")
