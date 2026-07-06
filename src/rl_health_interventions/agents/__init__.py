from __future__ import annotations

import logging

from rl_health_interventions._registry import Registry
from rl_health_interventions.agents._base import Agent

logger = logging.getLogger(__name__)

REGISTRY: Registry = Registry("agent")

_KNUTH_MULT = 2654435761  # Knuth multiplicative hash constant


def derive_agent_seed(base_seed: int, agent_index: int = 0) -> int:
    """Deterministic, independent agent seed from a base seed.

    Bandit agents ignore state, so correlation with environment RNG is harmless.
    State-aware agents (Phase 2) will need independent seed management.
    """
    return (base_seed * _KNUTH_MULT + agent_index) % (2**31)


def make(name: str, **kwargs) -> Agent:
    return REGISTRY.make(name, **kwargs)


# NOTE: Import new agent module above and append it here so register() runs on import.
# Each module must have a register() function that adds to REGISTRY.
from rl_health_interventions.agents import (  # noqa: E402
    contextual_bandits,
    fixed,
    random,
)

_AGENT_MODULES = [
    contextual_bandits,
    fixed,
    random,
]

REGISTRY.load_modules(_AGENT_MODULES, logger_name=__name__)
