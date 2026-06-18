"""Tests for ContextualBanditAgent base class (_base.py).

Covers _get_context_key routing, constructor parameter storage,
and error cases for contextual mode.
"""
from __future__ import annotations

import pytest

from rl_health_interventions.agents.contextual_bandits._base import (
    ContextualBanditAgent,
)
from rl_health_interventions.state import StateView


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ConcreteAgent(ContextualBanditAgent):
    """Minimal concrete subclass so we can instantiate the abstract base."""

    def select_action(self, state) -> str:  # type: ignore[override]
        return self._actions[0]

    def update(self, state, action: str, reward: float, next_state) -> None:
        pass


def _make_agent(**kwargs) -> _ConcreteAgent:
    return _ConcreteAgent(**kwargs)


# ---------------------------------------------------------------------------
# Constructor / attribute storage
# ---------------------------------------------------------------------------


def test_default_actions_when_none_passed():
    agent = _make_agent()
    assert agent._actions == ["nudge", "idle"]


def test_custom_actions_are_stored():
    agent = _make_agent(actions=["a", "b", "c"])
    assert agent._actions == ["a", "b", "c"]


def test_contextual_flag_stored_false_by_default():
    agent = _make_agent()
    assert agent.contextual is False


def test_contextual_flag_stored_true():
    agent = _make_agent(contextual=True, context_feature="activity")
    assert agent.contextual is True


def test_context_feature_stored_none_by_default():
    agent = _make_agent()
    assert agent.context_feature is None


def test_context_feature_stored_value():
    agent = _make_agent(contextual=True, context_feature="activity")
    assert agent.context_feature == "activity"


def test_rng_seeded_reproducibly():
    a1 = _make_agent(seed=7)
    a2 = _make_agent(seed=7)
    samples1 = [float(a1._rng.random()) for _ in range(20)]
    samples2 = [float(a2._rng.random()) for _ in range(20)]
    assert samples1 == samples2


def test_different_seeds_produce_different_rngs():
    a1 = _make_agent(seed=1)
    a2 = _make_agent(seed=2)
    samples1 = [float(a1._rng.random()) for _ in range(20)]
    samples2 = [float(a2._rng.random()) for _ in range(20)]
    assert samples1 != samples2


# ---------------------------------------------------------------------------
# _get_context_key – non-contextual mode
# ---------------------------------------------------------------------------


def test_get_context_key_non_contextual_returns_action_string():
    agent = _make_agent(contextual=False)
    state = StateView(activity="sedentary", day=0, step_of_day=0)
    key = agent._get_context_key(state, "nudge")
    assert key == "nudge"


def test_get_context_key_non_contextual_returns_idle_string():
    agent = _make_agent(contextual=False)
    state = StateView(activity="active", day=0, step_of_day=0)
    key = agent._get_context_key(state, "idle")
    assert key == "idle"


def test_get_context_key_non_contextual_ignores_state():
    """Context value is irrelevant in non-contextual mode."""
    agent = _make_agent(contextual=False)
    state_sed = StateView(activity="sedentary", day=0, step_of_day=0)
    state_act = StateView(activity="active", day=0, step_of_day=0)
    assert agent._get_context_key(state_sed, "nudge") == agent._get_context_key(
        state_act, "nudge"
    )


# ---------------------------------------------------------------------------
# _get_context_key – contextual mode
# ---------------------------------------------------------------------------


def test_get_context_key_contextual_returns_tuple():
    agent = _make_agent(contextual=True, context_feature="activity")
    state = StateView(activity="sedentary", day=0, step_of_day=0)
    key = agent._get_context_key(state, "nudge")
    assert key == ("sedentary", "nudge")


def test_get_context_key_contextual_active_context():
    agent = _make_agent(contextual=True, context_feature="activity")
    state = StateView(activity="active", day=0, step_of_day=0)
    key = agent._get_context_key(state, "idle")
    assert key == ("active", "idle")


def test_get_context_key_contextual_different_contexts_produce_different_keys():
    agent = _make_agent(contextual=True, context_feature="activity")
    sed = StateView(activity="sedentary", day=0, step_of_day=0)
    act = StateView(activity="active", day=0, step_of_day=0)
    assert agent._get_context_key(sed, "nudge") != agent._get_context_key(
        act, "nudge"
    )


def test_get_context_key_contextual_same_action_different_context():
    agent = _make_agent(contextual=True, context_feature="activity")
    state = StateView(activity="sedentary", day=0, step_of_day=0)
    key_nudge = agent._get_context_key(state, "nudge")
    key_idle = agent._get_context_key(state, "idle")
    assert key_nudge == ("sedentary", "nudge")
    assert key_idle == ("sedentary", "idle")


# ---------------------------------------------------------------------------
# _get_context_key – error cases
# ---------------------------------------------------------------------------


def test_get_context_key_contextual_none_feature_raises():
    """contextual=True but context_feature=None should raise ValueError."""
    agent = _make_agent(contextual=True, context_feature=None)
    state = StateView(activity="sedentary", day=0, step_of_day=0)
    with pytest.raises(ValueError, match="context_feature must be set"):
        agent._get_context_key(state, "nudge")


def test_get_context_key_state_missing_attribute_raises():
    """If state does not have the context_feature attribute, raise ValueError."""
    agent = _make_agent(contextual=True, context_feature="nonexistent_attr")
    state = StateView(activity="sedentary", day=0, step_of_day=0)
    with pytest.raises(ValueError, match="nonexistent_attr"):
        agent._get_context_key(state, "nudge")


def test_get_context_key_missing_attribute_error_message_lists_available():
    """Error message should mention available attributes."""
    agent = _make_agent(contextual=True, context_feature="missing_field")
    state = StateView(activity="sedentary", day=0, step_of_day=0)
    with pytest.raises(ValueError, match="Available attributes"):
        agent._get_context_key(state, "nudge")


def test_get_context_key_missing_attribute_error_includes_feature_name():
    agent = _make_agent(contextual=True, context_feature="bad_attr")
    state = StateView(activity="sedentary", day=0, step_of_day=0)
    with pytest.raises(ValueError, match="bad_attr"):
        agent._get_context_key(state, "nudge")