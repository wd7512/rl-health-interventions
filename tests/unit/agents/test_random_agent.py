from rl_health_interventions.agents.random import RandomAgent


def _sv() -> dict:
    return {"activity": "sedentary"}


def test_random_agent_returns_string_action():
    agent = RandomAgent(actions=["nudge", "idle"], seed=42)
    action = agent.select_action(_sv())
    assert action in ("nudge", "idle")


def test_random_agent_explores_both_actions():
    agent = RandomAgent(actions=["nudge", "idle"], seed=42)
    seen = set()
    for _ in range(100):
        seen.add(agent.select_action(_sv()))
    assert seen == {"nudge", "idle"}


def test_random_agent_update_is_noop():
    agent = RandomAgent(actions=["nudge", "idle"], seed=42)
    agent.update(_sv(), "nudge", 1.0, _sv())
