from __future__ import annotations

import logging

from rl_health_interventions.agents import (
    decaying_epsilon_greedy,
    epsilon_greedy,
    random,
    thompson_sampling,
    ucb,
)
from rl_health_interventions.agents._base import Agent

logger = logging.getLogger(__name__)

REGISTRY: dict[str, type[Agent]] = {}

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


# NOTE: Import new agent module above and append it here so register() runs on import.
# Each module must have a register() function that adds to REGISTRY.
_AGENT_MODULES = [
    thompson_sampling,
    epsilon_greedy,
    random,
    ucb,
    decaying_epsilon_greedy,
]

for _mod in _AGENT_MODULES:
    try:
        _mod.register()
    except Exception:
        logger.exception("Failed to register %s", _mod.__name__)
