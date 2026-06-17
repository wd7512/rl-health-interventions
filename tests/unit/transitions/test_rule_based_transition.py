from rl_health_interventions.transitions.rule_based import RuleBasedTransition


def test_transition_returns_valid_state(valid_config):
    t = RuleBasedTransition(valid_config, seed=42)
    next_state = t.transition("sedentary", "nudge")
    assert next_state in ("sedentary", "active")


def test_transition_probabilities_match_config_over_many_samples(valid_config):
    t = RuleBasedTransition(valid_config, seed=42)
    n_samples = 10_000
    n_active = sum(
        1 for _ in range(n_samples) if t.transition("sedentary", "nudge") == "active"
    )
    observed_freq = n_active / n_samples
    assert abs(observed_freq - 0.3) < 0.02
