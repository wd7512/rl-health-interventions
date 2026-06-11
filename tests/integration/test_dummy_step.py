from __future__ import annotations

from rl_health_interventions.transitions import make as make_transition
from rl_health_interventions.rewards import make as make_reward
from rl_health_interventions.agents import make as make_agent
from rl_health_interventions.simulation import make as make_response_model
from rl_health_interventions.data import make as make_dataset


def test_layer2_component_compatibility() -> None:
    """All registered components can be instantiated via the factory."""
    transition = make_transition("rule_based")
    reward = make_reward("compound")
    agent = make_agent("thompson_sampling")
    response = make_response_model("rule_based")
    dataset = make_dataset("synthetic")

    # Verify correct types (not just non-None)
    from rl_health_interventions.transitions._base import TransitionModel
    from rl_health_interventions.rewards._base import RewardHandler
    from rl_health_interventions.agents._base import Agent
    from rl_health_interventions.simulation._base import ResponseModel
    from rl_health_interventions.data.synthetic import SyntheticDataGenerator

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


def test_layer3_dummy_step() -> None:
    transition = make_transition("rule_based")
    reward = make_reward("compound")
    agent = make_agent("thompson_sampling")
    response = make_response_model("rule_based")

    dummy_state = {"step": 0}
    action = agent.select_action(dummy_state)
    agent.update(dummy_state, action, 0.0, dummy_state)
    next_state = transition.transition(dummy_state, action, None)
    rew, done = reward.reward(dummy_state, action, None)
    resp = response.response(dummy_state, action, None)
    assert isinstance(resp, float)
    assert isinstance(rew, float)
    assert isinstance(done, bool)
    assert next_state is dummy_state
