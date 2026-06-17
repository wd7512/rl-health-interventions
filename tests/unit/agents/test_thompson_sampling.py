from rl_health_interventions.agents.thompson_sampling import ThompsonSamplingAgent


def _sv() -> dict:
    return {"activity": "sedentary"}


def test_select_action_returns_string_action():
    agent = ThompsonSamplingAgent(actions=["nudge", "idle"], seed=42)
    action = agent.select_action(_sv())
    assert action in ("nudge", "idle")


def test_update_increases_alpha_on_positive_reward():
    agent = ThompsonSamplingAgent(
        actions=["nudge", "idle"], seed=42, alpha_prior=1.0, beta_prior=1.0
    )
    agent.update(_sv(), "nudge", reward=1.0, next_state=_sv())
    assert agent.posteriors["nudge"].alpha == 2.0
    assert agent.posteriors["nudge"].beta == 1.0


def test_update_increases_beta_on_zero_reward():
    agent = ThompsonSamplingAgent(
        actions=["nudge", "idle"], seed=42, alpha_prior=1.0, beta_prior=1.0
    )
    agent.update(_sv(), "idle", reward=0.0, next_state=_sv())
    assert agent.posteriors["idle"].alpha == 1.0
    assert agent.posteriors["idle"].beta == 2.0


def test_thompson_sampling_converges_to_better_arm():
    import numpy as np

    agent = ThompsonSamplingAgent(
        actions=["nudge", "idle"], seed=42, alpha_prior=1.0, beta_prior=1.0
    )
    arm_rewards = {"nudge": 0.8, "idle": 0.2}
    rng = np.random.default_rng(42)
    for _ in range(1000):
        action = agent.select_action(_sv())
        reward = 1.0 if rng.random() < arm_rewards[action] else 0.0
        agent.update(_sv(), action, reward, _sv())
    assert agent.posteriors["nudge"].alpha > agent.posteriors["nudge"].beta


def test_beta_arm_gets_more_beta_increments():
    import numpy as np

    agent = ThompsonSamplingAgent(
        actions=["nudge", "idle"], seed=42, alpha_prior=1.0, beta_prior=1.0
    )
    arm_rewards = {"nudge": 0.8, "idle": 0.2}
    rng = np.random.default_rng(42)
    for _ in range(1000):
        action = agent.select_action(_sv())
        reward = 1.0 if rng.random() < arm_rewards[action] else 0.0
        agent.update(_sv(), action, reward, _sv())
    better = agent.posteriors["nudge"]
    assert better.alpha > better.beta
    worse = agent.posteriors["idle"]
    better_total = better.alpha + better.beta
    worse_total = worse.alpha + worse.beta
    assert better_total > worse_total
