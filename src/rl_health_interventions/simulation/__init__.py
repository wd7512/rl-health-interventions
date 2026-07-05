from __future__ import annotations

import logging

from rl_health_interventions.simulation import rule_based
from rl_health_interventions.simulation._base import ResponseModel

logger = logging.getLogger(__name__)

REGISTRY: dict[str, type[ResponseModel]] = {}


def make(name: str, **kwargs) -> ResponseModel:
    if name not in REGISTRY:
        _msg = f"Unknown response model: {name}. Known: {list(REGISTRY)}"
        raise KeyError(_msg)
    return REGISTRY[name](**kwargs)


# NOTE: Import new simulation response model module above and append it here
# so register() runs on import. Each module must have a register()
# function that adds to REGISTRY.
_SIMULATION_MODULES = [rule_based]

for _mod in _SIMULATION_MODULES:
    try:
        _mod.register()
    except Exception:
        logger.exception("Failed to register %s", _mod.__name__)
