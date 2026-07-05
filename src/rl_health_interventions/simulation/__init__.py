from __future__ import annotations

from rl_health_interventions._registry import Registry
from rl_health_interventions.simulation._base import ResponseModel
from rl_health_interventions.simulation import rule_based
from rl_health_interventions.simulation._base import ResponseModel

REGISTRY: Registry = Registry("response model")


def make(name: str, **kwargs) -> ResponseModel:
    return REGISTRY.make(name, **kwargs)


# NOTE: Import new simulation response model module above and append it here
# so register() runs on import. Each module must have a register()
# function that adds to REGISTRY.
_SIMULATION_MODULES = [rule_based]

REGISTRY.load_modules(_SIMULATION_MODULES, logger_name=__name__)
