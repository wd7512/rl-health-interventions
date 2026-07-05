import pytest

from rl_health_interventions.agents.contextual_bandits.thompson_sampling import (
    ThompsonSamplingAgent,
)


def test_select_action_returns_string_action(state_view):
    agent = ThompsonSamplingAgent(actions=["nudge", "idle"], seed=42)
    action = agent.select_action(state_view)
    assert action in ("nudge", "idle")


@pytest.mark.parametrize(
    "action,reward,expected_alpha,expected_beta",
    [
        ("nudge", 1.0, 2.0, 1.0),
        ("idle", 0.0, 1.0, 2.0),
    ],
)
def test_update_updates_posterior(
    state_view, action, reward, expected_alpha, expected_beta
):
    agent = ThompsonSamplingAgent(
        actions=["nudge", "idle"], seed=42, alpha_prior=1.0, beta_prior=1.0
    )
    agent.update(state_view, action, reward=reward, next_state=state_view)
    assert agent.posteriors[action].alpha == expected_alpha
    assert agent.posteriors[action].beta == expected_beta


def test_thompson_sampling_converges_to_better_arm(state_view):
    import numpy as np

    agent = ThompsonSamplingAgent(
        actions=["nudge", "idle"], seed=42, alpha_prior=1.0, beta_prior=1.0
    )
    arm_rewards = {"nudge": 0.8, "idle": 0.2}
    rng = np.random.default_rng(42)
    for _ in range(1000):
        action = agent.select_action(state_view)
        reward = 1.0 if rng.random() < arm_rewards[action] else 0.0
        agent.update(state_view, action, reward, state_view)
    assert agent.posteriors["nudge"].alpha > agent.posteriors["nudge"].beta


def test_beta_arm_gets_more_beta_increments(state_view):
    import numpy as np

    agent = ThompsonSamplingAgent(
        actions=["nudge", "idle"], seed=42, alpha_prior=1.0, beta_prior=1.0
    )
    arm_rewards = {"nudge": 0.8, "idle": 0.2}
    rng = np.random.default_rng(42)
    for _ in range(1000):
        action = agent.select_action(state_view)
        reward = 1.0 if rng.random() < arm_rewards[action] else 0.0
        agent.update(state_view, action, reward, state_view)
    better = agent.posteriors["nudge"]
    assert better.alpha > better.beta
    worse = agent.posteriors["idle"]
    better_total = better.alpha + better.beta
    worse_total = worse.alpha + worse.beta
    assert better_total > worse_total


def test_agent_config_accepts_contextual_fields():
    from rl_health_interventions.config.schemas import AgentConfig

    config = AgentConfig(
        type="thompson_sampling",
        alpha_prior=1,
        beta_prior=1,
        contextual=True,
        context_features="activity_level",
    )
    assert config.contextual is True
    assert config.context_features == "activity_level"


def test_non_contextual_config_no_regression():
    from rl_health_interventions.config.schemas import AgentConfig

    config = AgentConfig(type="thompson_sampling", alpha_prior=1, beta_prior=1)
    assert config.contextual is False
    assert config.context_features is None


def test_contextual_ts_learns_context_dependent_optimal_actions():
    import numpy as np

    from rl_health_interventions.state import StateView

    agent = ThompsonSamplingAgent(
        actions=["nudge", "idle"],
        seed=42,
        alpha_prior=1.0,
        beta_prior=1.0,
        contextual=True,
        context_features="activity_level",
    )
    rng = np.random.default_rng(42)
    contexts = ["sedentary", "active"]
    n_iterations = 3000

    for _ in range(n_iterations):
        ctx = rng.choice(contexts)
        state = StateView(factors={"activity_level": ctx}, day=0, step_of_day=0)
        action = agent.select_action(state)
        if ctx == "sedentary":
            reward = 1.0 if rng.random() < (0.8 if action == "nudge" else 0.2) else 0.0
        else:
            reward = 1.0 if rng.random() < (0.2 if action == "nudge" else 0.8) else 0.0
        agent.update(state, action, reward, state)

    sed_nudge = agent.posteriors[("sedentary", "nudge")]
    sed_idle = agent.posteriors[("sedentary", "idle")]
    assert sed_nudge.alpha / (sed_nudge.alpha + sed_nudge.beta) > 0.6
    assert sed_idle.alpha / (sed_idle.alpha + sed_idle.beta) < 0.4

    act_nudge = agent.posteriors[("active", "nudge")]
    act_idle = agent.posteriors[("active", "idle")]
    assert act_nudge.alpha / (act_nudge.alpha + act_nudge.beta) < 0.4
    assert act_idle.alpha / (act_idle.alpha + act_idle.beta) > 0.6


def test_contextual_ts_uniform_rewards():
    import numpy as np

    from rl_health_interventions.state import StateView

    agent = ThompsonSamplingAgent(
        actions=["nudge", "idle"],
        seed=42,
        alpha_prior=1.0,
        beta_prior=1.0,
        contextual=True,
        context_features="activity_level",
    )
    rng = np.random.default_rng(42)
    contexts = ["sedentary", "active"]

    for _ in range(2000):
        ctx = rng.choice(contexts)
        state = StateView(factors={"activity_level": ctx}, day=0, step_of_day=0)
        action = agent.select_action(state)
        reward = 1.0 if rng.random() < 0.5 else 0.0
        agent.update(state, action, reward, state)

    for ctx in contexts:
        for act in ["nudge", "idle"]:
            p = agent.posteriors[(ctx, act)]
            ratio = p.alpha / (p.alpha + p.beta)
            assert 0.35 <= ratio <= 0.65, (
                f"({ctx}, {act}) ratio {ratio:.3f} out of range"
            )
