from __future__ import annotations

import logging
from typing import Type

from rl_health_interventions.agents._base import Agent
from rl_health_interventions.agents import epsilon_greedy
from rl_health_interventions.agents import random
from rl_health_interventions.agents import thompson_sampling
from rl_health_interventions.agents import ucb

logger = logging.getLogger(__name__)

REGISTRY: dict[str, Type[Agent]] = {}

_KNUTH_MULT = 2654435761  # Knuth multiplicative hash constant


def derive_agent_seed(base_seed: int, agent_index: int = 0) -> int:
    """Deterministic, independent agent seed from a base seed.

    Bandit agents ignore state, so correlation with environment RNG is harmless.
    State-aware agents (Phase 2) will need independent seed management.
    """
    return (base_seed * _KNUTH_MULT + agent_index) % (2**31)


def make(name: str, **kwargs) -> Agent:
    if name not in REGISTRY:
        raise KeyError(f"Unknown agent: {name}. Known: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)


try:
    thompson_sampling.register()
except Exception:
    logger.exception("Failed to register thompson_sampling agent")

try:
    epsilon_greedy.register()
except Exception:
    logger.exception("Failed to register epsilon_greedy agent")


try:
    random.register()
except Exception:
    logger.exception("Failed to register random agent")

try:
    ucb.register()
except Exception:
    logger.exception("Failed to register ucb agent")
