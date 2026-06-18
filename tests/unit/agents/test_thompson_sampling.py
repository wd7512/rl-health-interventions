import pytest

from rl_health_interventions.agents.thompson_sampling import ThompsonSamplingAgent


def test_select_action_returns_string_action(state_view):
    agent = ThompsonSamplingAgent(actions=["nudge", "idle"], seed=42)
    action = agent.select_action(state_view)
    assert action in ("nudge", "idle")


def test_update_increases_alpha_on_positive_reward(state_view):
    agent = ThompsonSamplingAgent(
        actions=["nudge", "idle"], seed=42, alpha_prior=1.0, beta_prior=1.0
    )
    agent.update(state_view, "nudge", reward=1.0, next_state=state_view)
    assert agent.posteriors["nudge"].alpha == 2.0
    assert agent.posteriors["nudge"].beta == 1.0


def test_update_increases_beta_on_zero_reward(state_view):
    agent = ThompsonSamplingAgent(
        actions=["nudge", "idle"], seed=42, alpha_prior=1.0, beta_prior=1.0
    )
    agent.update(state_view, "idle", reward=0.0, next_state=state_view)
    assert agent.posteriors["idle"].alpha == 1.0
    assert agent.posteriors["idle"].beta == 2.0


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
        context_feature="activity",
    )
    assert config.contextual is True
    assert config.context_feature == "activity"


def test_non_contextual_config_no_regression():
    from rl_health_interventions.config.schemas import AgentConfig

    config = AgentConfig(type="thompson_sampling", alpha_prior=1, beta_prior=1)
    assert config.contextual is False
    assert config.context_feature is None


def test_contextual_ts_learns_context_dependent_optimal_actions():
    import numpy as np
    from rl_health_interventions.state import StateView

    agent = ThompsonSamplingAgent(
        actions=["nudge", "idle"],
        seed=42,
        alpha_prior=1.0,
        beta_prior=1.0,
        contextual=True,
        context_feature="activity",
    )
    rng = np.random.default_rng(42)
    contexts = ["sedentary", "active"]
    n_iterations = 3000

    for _ in range(n_iterations):
        ctx = rng.choice(contexts)
        state = StateView(activity=ctx, day=0, step_of_day=0)
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
        context_feature="activity",
    )
    rng = np.random.default_rng(42)
    contexts = ["sedentary", "active"]

    for _ in range(2000):
        ctx = rng.choice(contexts)
        state = StateView(activity=ctx, day=0, step_of_day=0)
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


# ---------------------------------------------------------------------------
# Posterior NamedTuple
# ---------------------------------------------------------------------------


def test_posterior_has_alpha_and_beta_fields():
    from rl_health_interventions.agents.contextual_bandits.thompson_sampling import (
        Posterior,
    )

    p = Posterior(alpha=2.0, beta=3.0)
    assert p.alpha == 2.0
    assert p.beta == 3.0


def test_posterior_is_immutable():
    from rl_health_interventions.agents.contextual_bandits.thompson_sampling import (
        Posterior,
    )

    p = Posterior(alpha=1.0, beta=1.0)
    with pytest.raises(AttributeError):
        p.alpha = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Prior validation
# ---------------------------------------------------------------------------


def test_alpha_prior_zero_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        ThompsonSamplingAgent(alpha_prior=0.0, beta_prior=1.0)


def test_alpha_prior_negative_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        ThompsonSamplingAgent(alpha_prior=-1.0, beta_prior=1.0)


def test_beta_prior_zero_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        ThompsonSamplingAgent(alpha_prior=1.0, beta_prior=0.0)


def test_beta_prior_negative_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        ThompsonSamplingAgent(alpha_prior=1.0, beta_prior=-0.5)


# ---------------------------------------------------------------------------
# Parameter initialisation
# ---------------------------------------------------------------------------


def test_non_contextual_init_prepopulates_posteriors():
    agent = ThompsonSamplingAgent(
        actions=["nudge", "idle"], alpha_prior=2.0, beta_prior=3.0, seed=42
    )
    assert "nudge" in agent.posteriors
    assert "idle" in agent.posteriors
    assert agent.posteriors["nudge"].alpha == 2.0
    assert agent.posteriors["nudge"].beta == 3.0


def test_contextual_init_posteriors_start_empty():
    agent = ThompsonSamplingAgent(
        actions=["nudge", "idle"],
        alpha_prior=1.0,
        beta_prior=1.0,
        seed=42,
        contextual=True,
        context_feature="activity",
    )
    assert len(agent.posteriors) == 0


def test_ensure_params_creates_posterior_with_custom_priors():
    agent = ThompsonSamplingAgent(
        actions=["nudge", "idle"],
        alpha_prior=3.0,
        beta_prior=5.0,
        seed=42,
        contextual=True,
        context_feature="activity",
    )
    key = ("sedentary", "nudge")
    agent._ensure_params(key)
    assert key in agent.posteriors
    assert agent.posteriors[key].alpha == 3.0
    assert agent.posteriors[key].beta == 5.0


def test_ensure_params_is_idempotent():
    agent = ThompsonSamplingAgent(
        actions=["nudge", "idle"],
        alpha_prior=1.0,
        beta_prior=1.0,
        seed=42,
        contextual=True,
        context_feature="activity",
    )
    from rl_health_interventions.agents.contextual_bandits.thompson_sampling import (
        Posterior,
    )

    key = ("sedentary", "nudge")
    agent._ensure_params(key)
    # Manually set a different value
    agent.posteriors[key] = Posterior(alpha=10.0, beta=20.0)
    # Second call must not reset the value
    agent._ensure_params(key)
    assert agent.posteriors[key].alpha == 10.0


# ---------------------------------------------------------------------------
# Update routing
# ---------------------------------------------------------------------------


def test_contextual_update_routes_to_context_key(state_view):
    from rl_health_interventions.state import StateView

    agent = ThompsonSamplingAgent(
        actions=["nudge", "idle"],
        alpha_prior=1.0,
        beta_prior=1.0,
        seed=42,
        contextual=True,
        context_feature="activity",
    )
    state = StateView(activity="sedentary", day=0, step_of_day=0)
    agent.update(state, "nudge", reward=1.0, next_state=state)

    assert ("sedentary", "nudge") in agent.posteriors
    assert agent.posteriors[("sedentary", "nudge")].alpha == 2.0
    # raw action key must NOT be created
    assert "nudge" not in agent.posteriors


def test_non_contextual_update_routes_to_action_key(state_view):
    agent = ThompsonSamplingAgent(
        actions=["nudge", "idle"], alpha_prior=1.0, beta_prior=1.0, seed=42
    )
    agent.update(state_view, "nudge", reward=1.0, next_state=state_view)

    assert agent.posteriors["nudge"].alpha == 2.0
    # contextual tuple key must NOT be created
    assert ("sedentary", "nudge") not in agent.posteriors


# ---------------------------------------------------------------------------
# Regression: backward-compat import from old module path
# ---------------------------------------------------------------------------


def test_import_from_agents_module():
    """ThompsonSamplingAgent is still importable from the old top-level path."""
    from rl_health_interventions.agents.thompson_sampling import ThompsonSamplingAgent as TSA  # noqa: F401

    assert TSA is ThompsonSamplingAgent
