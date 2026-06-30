from __future__ import annotations

from pathlib import Path

from rl_health_interventions.agents import make as make_agent
from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.data import make as make_dataset
from rl_health_interventions.rewards import make as make_reward
from rl_health_interventions.simulation import make as make_response_model
from rl_health_interventions.state import StateView
from rl_health_interventions.transitions import make as make_transition

_HERE = Path(__file__).resolve().parent.parent
ASSETS_TABLES = str(_HERE / "assets" / "tables")


def _minimal_config() -> MDPConfig:
    return MDPConfig(
        episode_days=1,
        steps_per_day=1,
        seed=42,
        initial_state={"activity_level": "sedentary"},
        state={
            "factors": {
                "activity_level": {"dims": 2, "names": ["sedentary", "active"]},
            },
        },
        actions=["nudge", "idle"],
        reward={
            "factor": "activity_level",
            "values": {"sedentary": 0.0, "active": 1.0},
        },
        transition_model={"type": "rule_based", "table_dir": ASSETS_TABLES},
    )


def test_layer2_component_compatibility() -> None:
    config = _minimal_config()
    transition = make_transition(config)
    reward = make_reward(config)
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


def test_layer3_dummy_step() -> None:
    config = _minimal_config()
    transition = make_transition(config)
    reward = make_reward(config)
    agent = make_agent("thompson_sampling")
    response = make_response_model("rule_based")

    state = StateView({"activity_level": "sedentary"}, day=0, step_of_day=0)
    action = agent.select_action(state)
    agent.update(state, action, 0.0, state)
    next_state = transition.transition(state, action)
    rew, done = reward.reward(state, action, step_idx=0)
    resp = response.response(state, action)
    assert isinstance(resp, float)
    assert isinstance(rew, float)
    assert isinstance(done, bool)
    assert isinstance(next_state, StateView)
    assert next_state.activity_level in ("sedentary", "active")
