from rl_health_interventions.agents.fixed import FixedAgent


def test_fixed_agent_returns_configured_action():
    agent = FixedAgent(action="nudge")
    result = agent.select_action(None)
    assert result == "nudge"


def test_fixed_agent_defaults_to_idle():
    agent = FixedAgent()
    assert agent.select_action(None) == "idle"


def test_fixed_agent_ignores_state():
    agent = FixedAgent(action="idle")
    assert agent.select_action("any_state") == "idle"


def test_fixed_agent_update_is_noop():
    agent = FixedAgent(action="nudge")
    agent.update(None, "nudge", 1.0, None)
    assert agent.select_action(None) == "nudge"
