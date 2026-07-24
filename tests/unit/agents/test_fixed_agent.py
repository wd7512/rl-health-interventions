import pytest

from rl_health_interventions.agents.fixed import ComBWeightedFixedAgent, FixedAgent
from rl_health_interventions.config.schemas import AgentConfig


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


def test_fixed_agent_requires_action():
    with pytest.raises(ValueError, match="action must be provided"):
        AgentConfig.model_validate({"type": "fixed"})


@pytest.mark.parametrize("action", ["", "  ", 123])
def test_fixed_agent_rejects_invalid_action(action):
    with pytest.raises((ValueError, TypeError)):
        AgentConfig.model_validate({"type": "fixed", "action": action})


@pytest.mark.parametrize(
    "field",
    [
        {"alpha_prior": 1.0},
        {"beta_prior": 1.0},
        {"epsilon": 0.1},
        {"epsilon_start": 0.2},
        {"c": 2.0},
        {"contextual": True},
    ],
)
def test_fixed_agent_rejects_learning_params(field):
    with pytest.raises(ValueError, match="does not accept"):
        AgentConfig.model_validate({"type": "fixed", "action": "nudge", **field})


def test_comb_weighted_fixed_returns_composite_action():
    agent = ComBWeightedFixedAgent(
        comb_scores={
            "ability": 5,
            "perceived_benefit": 1,
            "planning": 5,
            "prioritization": 5,
            "social_opportunity": 5,
            "physical_opportunity": 5,
        },
        time_preference="morning",
        seed=1,
        actions=[
            "idle",
            "ability_morning",
            "ability_afternoon",
            "perceived_benefit_morning",
            "perceived_benefit_afternoon",
        ],
    )
    action = agent.select_action(None)
    assert action in {
        "ability_morning",
        "ability_afternoon",
        "perceived_benefit_morning",
        "perceived_benefit_afternoon",
    }


def test_comb_weighted_fixed_config_requires_persona_fields():
    with pytest.raises(ValueError, match="persona_comb_file"):
        AgentConfig.model_validate({"type": "comb_weighted_fixed"})


def test_comb_weighted_fixed_config_rejects_invalid_preference():
    with pytest.raises(ValueError, match="time_preference"):
        AgentConfig.model_validate(
            {
                "type": "comb_weighted_fixed",
                "persona_comb_file": "config/pearl/comb_scores.json",
                "persona_name": "base",
                "time_preference": "night",
            }
        )
