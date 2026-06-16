from rl_health_interventions.agents.ucb import UCBAgent
from rl_health_interventions.config.schemas import Action


def test_select_action_returns_valid_action():
    agent = UCBAgent(seed=42)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    action = agent.select_action(state)
    assert action in (Action.SEND, Action.DON_T_SEND)


def test_initial_exploration_pulls_each_action():
    """UCB pulls each action once before applying the UCB formula."""
    agent = UCBAgent(seed=42)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    # First two calls should pull each action exactly once
    first = agent.select_action(state)
    agent.update(state, first, 1.0, state)
    second = agent.select_action(state)
    agent.update(state, second, 0.0, state)
    assert first != second
    assert agent.counts[Action.SEND] == 1
    assert agent.counts[Action.DON_T_SEND] == 1


def test_prefers_better_arm_after_learning():
    agent = UCBAgent(c=1.0, seed=42)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    # Bootstrap: pull each arm once
    agent.update(state, Action.SEND, 1.0, state)
    agent.update(state, Action.DON_T_SEND, 0.0, state)
    # Pull both arms many times with different rewards
    for _ in range(100):
        agent.update(state, Action.SEND, 1.0, state)
    for _ in range(100):
        agent.update(state, Action.DON_T_SEND, 0.0, state)
    # SEND has Q=1.0, DON_T_SEND has Q=0.0, similar counts
    # UCB should always pick SEND
    for _ in range(20):
        assert agent.select_action(state) == Action.SEND


def test_ucb_exploration_bonus_decays():
    """After many pulls, UCB converges to the best Q-value arm."""
    agent = UCBAgent(c=2.0, seed=42)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    # Bootstrap
    agent.update(state, Action.SEND, 1.0, state)
    agent.update(state, Action.DON_T_SEND, 0.0, state)
    # Make SEND clearly better with many samples (tight confidence interval)
    for _ in range(200):
        agent.update(state, Action.SEND, 1.0, state)
    for _ in range(200):
        agent.update(state, Action.DON_T_SEND, 0.0, state)
    # With tight intervals, exploitation dominates
    for _ in range(20):
        assert agent.select_action(state) == Action.SEND


def test_seed_reproducibility():
    agent1 = UCBAgent(c=2.0, seed=99)
    agent2 = UCBAgent(c=2.0, seed=99)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    # Bootstrap both identically
    for a in agent1, agent2:
        a.update(state, Action.SEND, 1.0, state)
        a.update(state, Action.DON_T_SEND, 0.0, state)
    actions1 = [agent1.select_action(state) for _ in range(100)]
    actions2 = [agent2.select_action(state) for _ in range(100)]
    assert actions1 == actions2


def test_c_validation():
    import pytest

    with pytest.raises(ValueError, match="strictly positive"):
        UCBAgent(c=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        UCBAgent(c=-1.0)
