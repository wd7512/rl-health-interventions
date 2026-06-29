"""Tests for ContextualBanditAgent base class routing and error paths."""

from __future__ import annotations

import pytest
from rl_health_interventions.agents.thompson_sampling import ThompsonSamplingAgent
from rl_health_interventions.state import StateView


def _make_agent(**kwargs):
    return ThompsonSamplingAgent(
        actions=["nudge", "idle"], seed=42, alpha_prior=1.0, beta_prior=1.0, **kwargs
    )


def test_get_context_key_non_contextual_returns_action():
    agent = _make_agent(contextual=False)
    state = StateView({"activity": "sedentary"}, day=0, step_of_day=0)
    assert agent._get_context_key(state, "nudge") == "nudge"


def test_get_context_key_contextual_returns_tuple():
    agent = _make_agent(contextual=True, context_feature="activity")
    state = StateView({"activity": "sedentary"}, day=0, step_of_day=0)
    assert agent._get_context_key(state, "nudge") == ("sedentary", "nudge")


@pytest.mark.parametrize("invalid_feature", ["", "   ", 123, []])
def test_init_raises_when_contextual_with_invalid_context_feature(invalid_feature):
    with pytest.raises(ValueError, match="context_feature must be a non-empty string"):
        _make_agent(contextual=True, context_feature=invalid_feature)


def test_get_context_key_contextual_raises_on_none_state():
    agent = _make_agent(contextual=True, context_feature="activity")
    with pytest.raises(ValueError, match="state cannot be None"):
        agent._get_context_key(None, "nudge")


def test_get_context_key_contextual_raises_on_missing_attribute():
    agent = _make_agent(contextual=True, context_feature="nonexistent")
    state = StateView({"activity": "sedentary"}, day=0, step_of_day=0)
    with pytest.raises(AttributeError):
        agent._get_context_key(state, "nudge")
