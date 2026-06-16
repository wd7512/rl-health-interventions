from __future__ import annotations

import pytest

from rl_health_interventions.agents import REGISTRY, make
from rl_health_interventions.agents.thompson_sampling import ThompsonSamplingAgent
from rl_health_interventions.agents.epsilon_greedy import EpsilonGreedyAgent
from rl_health_interventions.agents.random import RandomAgent
from rl_health_interventions.agents.ucb import UCBAgent


def test_all_agents_registered():
    assert "thompson_sampling" in REGISTRY
    assert "epsilon_greedy" in REGISTRY
    assert "random" in REGISTRY
    assert "ucb" in REGISTRY


def test_make_thompson_sampling():
    instance = make("thompson_sampling")
    assert isinstance(instance, ThompsonSamplingAgent)


def test_make_epsilon_greedy():
    instance = make("epsilon_greedy")
    assert isinstance(instance, EpsilonGreedyAgent)


def test_make_random():
    instance = make("random")
    assert isinstance(instance, RandomAgent)


def test_make_ucb():
    instance = make("ucb")
    assert isinstance(instance, UCBAgent)


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
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        EpsilonGreedyAgent(epsilon=1.5)
