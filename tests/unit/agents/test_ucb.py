from rl_health_interventions.agents.ucb import UCBAgent
from rl_health_interventions.state import StateView


def _sv() -> StateView:
    return StateView(activity="sedentary", day=0, step_of_day=0)


def test_select_action_returns_string_action():
    agent = UCBAgent(actions=["nudge", "idle"], seed=42)
    action = agent.select_action(_sv())
    assert action in ("nudge", "idle")


def test_initial_exploration_pulls_each_action():
    agent = UCBAgent(actions=["nudge", "idle"], seed=42)
    state = _sv()
    first = agent.select_action(state)
    agent.update(state, first, 1.0, state)
    second = agent.select_action(state)
    agent.update(state, second, 0.0, state)
    assert first != second
    assert agent.counts["nudge"] == 1
    assert agent.counts["idle"] == 1


def test_prefers_better_arm_after_learning():
    agent = UCBAgent(actions=["nudge", "idle"], c=1.0, seed=42)
    state = _sv()
    agent.update(state, "nudge", 1.0, state)
    agent.update(state, "idle", 0.0, state)
    for _ in range(100):
        agent.update(state, "nudge", 1.0, state)
    for _ in range(100):
        agent.update(state, "idle", 0.0, state)
    for _ in range(20):
        assert agent.select_action(state) == "nudge"


def test_ucb_exploration_bonus_decays():
    agent = UCBAgent(actions=["nudge", "idle"], c=2.0, seed=42)
    state = _sv()
    agent.update(state, "nudge", 1.0, state)
    agent.update(state, "idle", 0.0, state)
    for _ in range(200):
        agent.update(state, "nudge", 1.0, state)
    for _ in range(200):
        agent.update(state, "idle", 0.0, state)
    for _ in range(20):
        assert agent.select_action(state) == "nudge"


def test_seed_reproducibility():
    agent1 = UCBAgent(actions=["nudge", "idle"], c=2.0, seed=99)
    agent2 = UCBAgent(actions=["nudge", "idle"], c=2.0, seed=99)
    state = _sv()
    for a in agent1, agent2:
        a.update(state, "nudge", 1.0, state)
        a.update(state, "idle", 0.0, state)
    actions1 = [agent1.select_action(state) for _ in range(100)]
    actions2 = [agent2.select_action(state) for _ in range(100)]
    assert actions1 == actions2


def test_c_validation():
    import pytest

    with pytest.raises(ValueError, match="strictly positive"):
        UCBAgent(c=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        UCBAgent(c=-1.0)
