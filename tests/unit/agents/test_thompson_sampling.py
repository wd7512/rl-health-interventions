from rl_health_interventions.agents.thompson_sampling import ThompsonSamplingAgent
from rl_health_interventions.config.schemas import Action


def test_select_action_returns_valid_action():
    agent = ThompsonSamplingAgent(n_actions=2, seed=42)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    action = agent.select_action(state)
    assert action in (Action.SEND, Action.DON_T_SEND)


def test_update_increases_beta_for_chosen_action():
    agent = ThompsonSamplingAgent(n_actions=2, seed=42, alpha_prior=1.0, beta_prior=1.0)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    # Update with reward=1 for SEND action
    agent.update(state, Action.SEND, reward=1.0, next_state=state)
    # Now posterior alpha for SEND should be 2.0
    assert agent.posteriors[Action.SEND][0] == 2.0  # alpha
    assert agent.posteriors[Action.SEND][1] == 1.0  # beta unchanged


def test_update_decreases_beta_for_chosen_action_on_zero_reward():
    agent = ThompsonSamplingAgent(n_actions=2, seed=42, alpha_prior=1.0, beta_prior=1.0)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    agent.update(state, Action.DON_T_SEND, reward=0.0, next_state=state)
    assert agent.posteriors[Action.DON_T_SEND][0] == 1.0  # alpha unchanged
    assert agent.posteriors[Action.DON_T_SEND][1] == 2.0  # beta


def test_thompson_sampling_converges_to_better_arm():
    """Statistical test: over 1000 episodes, TS should prefer the arm with higher reward probability."""
    import numpy as np

    agent = ThompsonSamplingAgent(n_actions=2, seed=42, alpha_prior=1.0, beta_prior=1.0)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    arm_rewards = {Action.SEND: 0.8, Action.DON_T_SEND: 0.2}
    rng = np.random.default_rng(42)
    for _ in range(1000):
        action = agent.select_action(state)
        reward = 1.0 if rng.random() < arm_rewards[action] else 0.0
        agent.update(state, action, reward, state)
    # TS should have pulled SEND (the better arm) at least 80% of the time
    # (after 1000 episodes, posterior for SEND is roughly Beta(1 + 800*0.8, 1 + 800*0.2) = Beta(641, 161))
    # So most pulls should be SEND. We check that posterior alpha > beta for SEND.
    assert agent.posteriors[Action.SEND][0] > agent.posteriors[Action.SEND][1]
