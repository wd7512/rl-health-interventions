from rl_health_interventions.config.schemas import (
    ActivityLevel, Action, TimeOfDay, MDPConfig, TransitionMatrix, TimeOfDayMask
)
from rl_health_interventions.transitions.rule_based import RuleBasedTransition


def _config() -> MDPConfig:
    return MDPConfig(
        activity_levels=[ActivityLevel.SEDENTARY, ActivityLevel.ACTIVE],
        actions=[Action.SEND, Action.DON_T_SEND],
        time_of_day=[TimeOfDay.MORNING, TimeOfDay.MIDDAY, TimeOfDay.AFTERNOON, TimeOfDay.EVENING, TimeOfDay.NIGHT],
        transition=TransitionMatrix(root={
            ActivityLevel.SEDENTARY: {
                Action.SEND: {ActivityLevel.SEDENTARY: 0.7, ActivityLevel.ACTIVE: 0.3},
                Action.DON_T_SEND: {ActivityLevel.SEDENTARY: 0.9, ActivityLevel.ACTIVE: 0.1},
            },
            ActivityLevel.ACTIVE: {
                Action.SEND: {ActivityLevel.SEDENTARY: 0.2, ActivityLevel.ACTIVE: 0.8},
                Action.DON_T_SEND: {ActivityLevel.SEDENTARY: 0.4, ActivityLevel.ACTIVE: 0.6},
            },
        }),
        masks=TimeOfDayMask(root={
            TimeOfDay.MORNING: {ActivityLevel.SEDENTARY: 0.0, ActivityLevel.ACTIVE: 0.0},
            TimeOfDay.MIDDAY: {ActivityLevel.SEDENTARY: 0.0, ActivityLevel.ACTIVE: 0.0},
            TimeOfDay.AFTERNOON: {ActivityLevel.SEDENTARY: 0.0, ActivityLevel.ACTIVE: 0.0},
            TimeOfDay.EVENING: {ActivityLevel.SEDENTARY: 0.0, ActivityLevel.ACTIVE: 0.0},
            TimeOfDay.NIGHT: {ActivityLevel.SEDENTARY: 1.0, ActivityLevel.ACTIVE: 1.0},
        }),
    )


def test_transition_returns_valid_state():
    t = RuleBasedTransition(_config(), seed=42)
    next_state = t.transition(ActivityLevel.SEDENTARY, Action.SEND, TimeOfDay.MORNING)
    assert next_state in (ActivityLevel.SEDENTARY, ActivityLevel.ACTIVE)


def test_transition_night_mask_forces_no_change():
    t = RuleBasedTransition(_config(), seed=42)
    for _ in range(100):
        next_state = t.transition(ActivityLevel.SEDENTARY, Action.SEND, TimeOfDay.NIGHT)
        assert next_state == ActivityLevel.SEDENTARY
        next_state = t.transition(ActivityLevel.ACTIVE, Action.DON_T_SEND, TimeOfDay.NIGHT)
        assert next_state == ActivityLevel.ACTIVE


def test_transition_probabilities_match_config_over_many_samples():
    t = RuleBasedTransition(_config(), seed=42)
    n_samples = 10_000
    n_active = sum(
        1 for _ in range(n_samples)
        if t.transition(ActivityLevel.SEDENTARY, Action.SEND, TimeOfDay.MORNING) == ActivityLevel.ACTIVE
    )
    observed_freq = n_active / n_samples
    assert abs(observed_freq - 0.3) < 0.02
