from __future__ import annotations

import pytest

from rl_health_interventions.agents import REGISTRY, make
from rl_health_interventions.agents.contextual_bandits.epsilon_greedy import (
    EpsilonGreedyAgent,
)
from rl_health_interventions.agents.contextual_bandits.thompson_sampling import (
    ThompsonSamplingAgent,
)
from rl_health_interventions.agents.contextual_bandits.ucb import UCBAgent
from rl_health_interventions.agents.fixed import FixedAgent
from rl_health_interventions.agents.random import RandomAgent


def test_all_agents_registered():
    assert "thompson_sampling" in REGISTRY
    assert "epsilon_greedy" in REGISTRY
    assert "random" in REGISTRY
    assert "ucb" in REGISTRY
    assert "fixed" in REGISTRY


@pytest.mark.parametrize(
    ("agent_type", "expected_cls"),
    [
        ("thompson_sampling", ThompsonSamplingAgent),
        ("epsilon_greedy", EpsilonGreedyAgent),
        ("random", RandomAgent),
        ("ucb", UCBAgent),
        ("fixed", FixedAgent),
    ],
)
def test_make_agent(agent_type, expected_cls):
    assert isinstance(make(agent_type), expected_cls)


@pytest.mark.parametrize(
    ("agent_type", "kwargs", "check_attr", "expected_val"),
    [
        ("epsilon_greedy", {"epsilon": 0.5}, "epsilon", 0.5),
        (
            "thompson_sampling",
            {"alpha_prior": 2.0, "beta_prior": 3.0},
            "alpha_prior",
            2.0,
        ),
        ("ucb", {"c": 1.5}, "c", 1.5),
        ("fixed", {"action": "nudge"}, "_action", "nudge"),
    ],
)
def test_make_with_kwargs(agent_type, kwargs, check_attr, expected_val):
    agent = make(agent_type, **kwargs)
    assert getattr(agent, check_attr) == expected_val


def test_make_unknown_raises_keyerror():
    with pytest.raises(KeyError, match="NonExistent"):
        make("NonExistent")


def test_make_with_unexpected_kwargs_raises():
    with pytest.raises(TypeError):
        make("random", nonexistent=123)


def test_ts_negative_prior_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        ThompsonSamplingAgent(alpha_prior=-1.0)


def test_eg_out_of_range_epsilon_raises():
    with pytest.raises(ValueError, match=r"between 0.0 and 1.0"):
        EpsilonGreedyAgent(epsilon=1.5)
