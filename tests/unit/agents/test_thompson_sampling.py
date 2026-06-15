from rl_health_interventions.agents.thompson_sampling import ThompsonSamplingAgent
from rl_health_interventions.config.schemas import Action, ActivityLevel, TimeOfDay
from rl_health_interventions.state import StateView


def _sv(activity: ActivityLevel = ActivityLevel.SEDENTARY) -> StateView:
    return StateView(activity, TimeOfDay.MORNING, day=0, step_of_day=0)


def test_select_action_returns_valid_action():
    agent = ThompsonSamplingAgent(n_actions=2, seed=42)
    action = agent.select_action(_sv())
    assert action in (Action.SEND, Action.DON_T_SEND)


def test_update_increases_alpha_when_next_state_is_active():
    agent = ThompsonSamplingAgent(n_actions=2, seed=42, alpha_prior=1.0, beta_prior=1.0)
    agent.update(_sv(), Action.SEND, reward=1.0, next_state=_sv(ActivityLevel.ACTIVE))
    assert agent.posteriors[Action.SEND][0] == 2.0  # alpha
    assert agent.posteriors[Action.SEND][1] == 1.0  # beta unchanged


def test_update_increases_beta_when_next_state_is_sedentary():
    agent = ThompsonSamplingAgent(n_actions=2, seed=42, alpha_prior=1.0, beta_prior=1.0)
    agent.update(_sv(), Action.DON_T_SEND, reward=0.0, next_state=_sv(ActivityLevel.SEDENTARY))
    assert agent.posteriors[Action.DON_T_SEND][0] == 1.0  # alpha unchanged
    assert agent.posteriors[Action.DON_T_SEND][1] == 2.0  # beta


def test_thompson_sampling_converges_to_better_arm():
    """Statistical test: over 1000 episodes, TS should prefer the arm with higher reward probability."""
    import numpy as np

    agent = ThompsonSamplingAgent(n_actions=2, seed=42, alpha_prior=1.0, beta_prior=1.0)
    arm_rewards = {Action.SEND: 0.8, Action.DON_T_SEND: 0.2}
    rng = np.random.default_rng(42)
    for _ in range(1000):
        action = agent.select_action(_sv())
        is_active = rng.random() < arm_rewards[action]
        next_sv = StateView(
            ActivityLevel.ACTIVE if is_active else ActivityLevel.SEDENTARY,
            TimeOfDay.MORNING, day=0, step_of_day=0,
        )
        agent.update(_sv(), action, 1.0 if is_active else 0.0, next_sv)
    # TS should have pulled SEND (the better arm) more often
    assert agent.posteriors[Action.SEND][0] > agent.posteriors[Action.SEND][1]
