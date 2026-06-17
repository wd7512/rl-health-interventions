from rl_health_interventions.agents.random import RandomAgent


def test_random_agent_returns_string_action(state_view):
    agent = RandomAgent(actions=["nudge", "idle"], seed=42)
    action = agent.select_action(state_view)
    assert action in ("nudge", "idle")


def test_random_agent_explores_both_actions(state_view):
    agent = RandomAgent(actions=["nudge", "idle"], seed=42)
    seen = set()
    for _ in range(100):
        seen.add(agent.select_action(state_view))
    assert seen == {"nudge", "idle"}


def test_random_agent_update_is_noop(state_view):
    agent = RandomAgent(actions=["nudge", "idle"], seed=42)
    agent.update(state_view, "nudge", 1.0, state_view)
    # Random agent has no internal state to mutate — verify it still works
    action_after = agent.select_action(state_view)
    assert action_after in ("nudge", "idle")
