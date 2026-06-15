from rl_health_interventions.agents.random import RandomAgent
from rl_health_interventions.config.schemas import Action, ActivityLevel, TimeOfDay
from rl_health_interventions.state import StateView


def _sv() -> StateView:
    return StateView(ActivityLevel.SEDENTARY, TimeOfDay.MORNING, day=0, step_of_day=0)


def test_random_agent_returns_valid_action():
    agent = RandomAgent(seed=42)
    action = agent.select_action(_sv())
    assert action in (Action.SEND, Action.DON_T_SEND)


def test_random_agent_explores_both_actions():
    """Over many calls, both actions should appear."""
    agent = RandomAgent(seed=42)
    seen = set()
    for _ in range(100):
        seen.add(agent.select_action(_sv()))
    assert seen == {Action.SEND, Action.DON_T_SEND}


def test_random_agent_update_is_noop():
    agent = RandomAgent(seed=42)
    # Should not raise
    agent.update(_sv(), Action.SEND, 1.0, _sv())
