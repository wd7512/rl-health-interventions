from rl_health_interventions.agents.thompson_sampling import ThompsonSamplingAgent
from rl_health_interventions.config.schemas import Action, ActivityLevel, TimeOfDay
from rl_health_interventions.state import StateView


def _sv(activity: ActivityLevel = ActivityLevel.SEDENTARY) -> StateView:
    return StateView(activity, TimeOfDay.MORNING, day=0, step_of_day=0, steps_per_day=5)


def test_select_action_returns_valid_action():
    agent = ThompsonSamplingAgent(seed=42)
    action = agent.select_action(_sv())
    assert action in (Action.SEND, Action.DON_T_SEND)


def test_update_increases_alpha_on_positive_reward():
    agent = ThompsonSamplingAgent(seed=42, alpha_prior=1.0, beta_prior=1.0)
    agent.update(_sv(), Action.SEND, reward=1.0, next_state=_sv())
    assert agent.posteriors[Action.SEND].alpha == 2.0
    assert agent.posteriors[Action.SEND].beta == 1.0


def test_update_increases_beta_on_zero_reward():
    agent = ThompsonSamplingAgent(seed=42, alpha_prior=1.0, beta_prior=1.0)
    agent.update(_sv(), Action.DON_T_SEND, reward=0.0, next_state=_sv())
    assert agent.posteriors[Action.DON_T_SEND].alpha == 1.0
    assert agent.posteriors[Action.DON_T_SEND].beta == 2.0


def test_thompson_sampling_converges_to_better_arm():
    import numpy as np

    agent = ThompsonSamplingAgent(seed=42, alpha_prior=1.0, beta_prior=1.0)
    arm_rewards = {Action.SEND: 0.8, Action.DON_T_SEND: 0.2}
    rng = np.random.default_rng(42)
    for _ in range(1000):
        action = agent.select_action(_sv())
        reward = 1.0 if rng.random() < arm_rewards[action] else 0.0
        agent.update(_sv(), action, reward, _sv())
    assert agent.posteriors[Action.SEND].alpha > agent.posteriors[Action.SEND].beta


def test_beta_arm_gets_more_beta_increments():
    import numpy as np

    agent = ThompsonSamplingAgent(seed=42, alpha_prior=1.0, beta_prior=1.0)
    arm_rewards = {Action.SEND: 0.8, Action.DON_T_SEND: 0.2}
    rng = np.random.default_rng(42)
    for _ in range(1000):
        action = agent.select_action(_sv())
        reward = 1.0 if rng.random() < arm_rewards[action] else 0.0
        agent.update(_sv(), action, reward, _sv())
    # The better arm should have more alpha than beta (net positive)
    better = agent.posteriors[Action.SEND]
    assert better.alpha > better.beta
    # The worse arm should have fewer total increments than the better arm
    # (it's pulled less often because TS converges away from it)
    worse = agent.posteriors[Action.DON_T_SEND]
    better_total = better.alpha + better.beta
    worse_total = worse.alpha + worse.beta
    assert better_total > worse_total
