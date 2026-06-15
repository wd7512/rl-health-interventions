from rl_health_interventions.config.schemas import (
    ActivityLevel,
    Action,
    TimeOfDay,
)
from rl_health_interventions.transitions.rule_based import RuleBasedTransition


def test_transition_returns_valid_state(valid_config):
    t = RuleBasedTransition(valid_config, seed=42)
    next_state = t.transition(ActivityLevel.SEDENTARY, Action.SEND, TimeOfDay.MORNING)
    assert next_state in (ActivityLevel.SEDENTARY, ActivityLevel.ACTIVE)


def test_transition_night_mask_forces_no_change(valid_config):
    t = RuleBasedTransition(valid_config, seed=42)
    for _ in range(100):
        next_state = t.transition(ActivityLevel.SEDENTARY, Action.SEND, TimeOfDay.NIGHT)
        assert next_state == ActivityLevel.SEDENTARY
        next_state = t.transition(
            ActivityLevel.ACTIVE, Action.DON_T_SEND, TimeOfDay.NIGHT
        )
        assert next_state == ActivityLevel.ACTIVE


def test_transition_probabilities_match_config_over_many_samples(valid_config):
    t = RuleBasedTransition(valid_config, seed=42)
    n_samples = 10_000
    n_active = sum(
        1
        for _ in range(n_samples)
        if t.transition(ActivityLevel.SEDENTARY, Action.SEND, TimeOfDay.MORNING)
        == ActivityLevel.ACTIVE
    )
    observed_freq = n_active / n_samples
    assert abs(observed_freq - 0.3) < 0.02
