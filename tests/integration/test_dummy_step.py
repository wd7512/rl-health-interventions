from __future__ import annotations

import logging

from rl_health_interventions.transitions import make as make_transition
from rl_health_interventions.rewards import make as make_reward
from rl_health_interventions.agents import make as make_agent
from rl_health_interventions.simulation import make as make_response_model
from rl_health_interventions.data import make as make_dataset

logger = logging.getLogger(__name__)


def test_layer2_component_compatibility() -> None:
    transition = make_transition("RuleBasedTransition")
    reward = make_reward("CompoundReward")
    agent = make_agent("ThompsonSamplingAgent")
    response = make_response_model("RuleBasedResponse")
    dataset = make_dataset("SyntheticDataGenerator")

    assert transition is not None
    assert reward is not None
    assert agent is not None
    assert response is not None
    assert dataset is not None


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
    transition = make_transition("RuleBasedTransition")
    reward = make_reward("CompoundReward")
    agent = make_agent("ThompsonSamplingAgent")
    response = make_response_model("RuleBasedResponse")

    dummy_state = {"step": 0}
    action = agent.select_action(dummy_state)
    agent.update(dummy_state, action, 0.0, dummy_state)
    next_state = transition.transition(dummy_state, action, None)
    rew, done = reward.reward(dummy_state, action, None)
    response.response(dummy_state, action, None)

    assert isinstance(action, int)
    assert isinstance(rew, float)
    assert isinstance(done, bool)
    assert next_state is dummy_state
