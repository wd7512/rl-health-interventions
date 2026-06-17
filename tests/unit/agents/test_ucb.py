from rl_health_interventions.agents.ucb import UCBAgent


def test_select_action_returns_string_action(state_view):
    agent = UCBAgent(actions=["nudge", "idle"], seed=42)
    action = agent.select_action(state_view)
    assert action in ("nudge", "idle")


def test_initial_exploration_pulls_each_action(state_view):
    agent = UCBAgent(actions=["nudge", "idle"], seed=42)
    first = agent.select_action(state_view)
    agent.update(state_view, first, 1.0, state_view)
    second = agent.select_action(state_view)
    agent.update(state_view, second, 0.0, state_view)
    assert first != second
    assert agent.counts["nudge"] == 1
    assert agent.counts["idle"] == 1


def test_prefers_better_arm_after_learning(state_view):
    agent = UCBAgent(actions=["nudge", "idle"], c=1.0, seed=42)
    agent.update(state_view, "nudge", 1.0, state_view)
    agent.update(state_view, "idle", 0.0, state_view)
    for _ in range(100):
        agent.update(state_view, "nudge", 1.0, state_view)
    for _ in range(100):
        agent.update(state_view, "idle", 0.0, state_view)
    for _ in range(20):
        assert agent.select_action(state_view) == "nudge"


def test_ucb_exploration_bonus_decays(state_view):
    agent = UCBAgent(actions=["nudge", "idle"], c=2.0, seed=42)
    agent.update(state_view, "nudge", 1.0, state_view)
    agent.update(state_view, "idle", 0.0, state_view)
    for _ in range(200):
        agent.update(state_view, "nudge", 1.0, state_view)
    for _ in range(200):
        agent.update(state_view, "idle", 0.0, state_view)
    for _ in range(20):
        assert agent.select_action(state_view) == "nudge"


def test_seed_reproducibility(state_view):
    agent1 = UCBAgent(actions=["nudge", "idle"], c=2.0, seed=99)
    agent2 = UCBAgent(actions=["nudge", "idle"], c=2.0, seed=99)
    for a in agent1, agent2:
        a.update(state_view, "nudge", 1.0, state_view)
        a.update(state_view, "idle", 0.0, state_view)
    actions1 = [agent1.select_action(state_view) for _ in range(100)]
    actions2 = [agent2.select_action(state_view) for _ in range(100)]
    assert actions1 == actions2


def test_c_validation():
    import pytest

    with pytest.raises(ValueError, match="strictly positive"):
        UCBAgent(c=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        UCBAgent(c=-1.0)
