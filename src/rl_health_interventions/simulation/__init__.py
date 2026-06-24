from __future__ import annotations

import logging
from typing import Type

from rl_health_interventions.simulation._base import ResponseModel
from rl_health_interventions.simulation import rule_based

logger = logging.getLogger(__name__)

REGISTRY: dict[str, Type[ResponseModel]] = {}


def make(name: str, **kwargs) -> ResponseModel:
    if name not in REGISTRY:
        raise KeyError(f"Unknown response model: {name}. Known: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)


_SIMULATION_MODULES = [rule_based]

for _mod in _SIMULATION_MODULES:
    try:
        _mod.register()
    except Exception:
        logger.exception("Failed to register %s", _mod.__name__)
