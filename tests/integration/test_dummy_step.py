from __future__ import annotations

from rl_health_interventions.agents import make as make_agent
from rl_health_interventions.data import make as make_dataset
from rl_health_interventions.rewards import make as make_reward
from rl_health_interventions.simulation import make as make_response_model
from rl_health_interventions.transitions import make as make_transition


def test_layer2_component_compatibility(minimal_config) -> None:
    transition = make_transition(minimal_config)
    reward = make_reward(minimal_config)
    agent = make_agent("thompson_sampling")
    response = make_response_model("rule_based")
    dataset = make_dataset("synthetic")

    from rl_health_interventions.agents._base import Agent
    from rl_health_interventions.data.synthetic import SyntheticDataGenerator
    from rl_health_interventions.rewards._base import RewardHandler
    from rl_health_interventions.simulation._base import ResponseModel
    from rl_health_interventions.transitions._base import TransitionModel

    assert isinstance(transition, TransitionModel)
    assert isinstance(reward, RewardHandler)
    assert isinstance(agent, Agent)
    assert isinstance(response, ResponseModel)
    assert isinstance(dataset, SyntheticDataGenerator)


def test_layer2_unknown_component_fails() -> None:
    import pytest

    with pytest.raises(KeyError):
        make_transition("DoesNotExist")
    with pytest.raises(KeyError):
        make_reward("DoesNotExist")
    with pytest.raises(KeyError):
        make_agent("DoesNotExist")
    with pytest.raises(KeyError):
        make_response_model("DoesNotExist")
    with pytest.raises(KeyError):
        make_dataset("DoesNotExist")


def test_layer3_dummy_step(minimal_config) -> None:
    from rl_health_interventions.state import StateView

    transition = make_transition(minimal_config)
    reward = make_reward(minimal_config)
    agent = make_agent("thompson_sampling")
    response = make_response_model("rule_based")

    state = StateView(factors={"activity_level": "sedentary"}, day=0, step_of_day=0)
    action = agent.select_action(state)
    agent.update(state, action, 0.0, state)
    next_updates = transition.transition(state, action)
    rew, done = reward.reward(state, action, step_idx=0)
    resp = response.response(state, action)
    assert isinstance(resp, float)
    assert isinstance(rew, float)
    assert isinstance(done, bool)
    assert next_updates["activity_level"] in ("sedentary", "active")


def test_fixed_agent_layer2_compatibility() -> None:
    from rl_health_interventions.agents._base import Agent

    agent = make_agent("fixed", action="nudge")
    assert isinstance(agent, Agent)
    assert agent.select_action(None) == "nudge"


def test_fixed_agent_layer3_dummy_step(minimal_config) -> None:
    agent = make_agent("fixed", action="nudge", actions=["nudge", "idle"])
    state = "sedentary"
    action = agent.select_action(state)
    assert action == "nudge"
    agent.update(state, action, 1.0, "active")
    assert agent.select_action(state) == "nudge"
