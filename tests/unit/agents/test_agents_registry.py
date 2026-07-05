from __future__ import annotations

import pytest

from rl_health_interventions.agents import REGISTRY, make
from rl_health_interventions.agents.epsilon_greedy import EpsilonGreedyAgent
from rl_health_interventions.agents.random import RandomAgent
from rl_health_interventions.agents.thompson_sampling import ThompsonSamplingAgent
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
    with pytest.raises(ValueError, match=r"between 0.0 and 1.0"):
        EpsilonGreedyAgent(epsilon=1.5)


def test_make_passes_kwargs_to_constructor():
    agent = make("epsilon_greedy", epsilon=0.5, seed=99)
    assert isinstance(agent, EpsilonGreedyAgent)
    assert agent.epsilon == 0.5


def test_make_ts_with_explicit_priors():
    agent = make("thompson_sampling", alpha_prior=2.0, beta_prior=3.0, seed=42)
    assert isinstance(agent, ThompsonSamplingAgent)
    assert agent.alpha_prior == 2.0
    assert agent.beta_prior == 3.0


def test_make_ucb_with_custom_c():
    agent = make("ucb", c=1.5, seed=42)
    assert isinstance(agent, UCBAgent)
    assert agent.c == 1.5
