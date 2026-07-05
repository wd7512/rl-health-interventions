from __future__ import annotations

from rl_health_interventions._registry import Registry
from rl_health_interventions.data import synthetic, feature_pipeline

REGISTRY: Registry = Registry("data component")


def make(name: str, **kwargs) -> object:
    return REGISTRY.make(name, **kwargs)


_DATA_COMPONENT_MODULES = [synthetic, feature_pipeline]

REGISTRY.load_modules(_DATA_COMPONENT_MODULES, logger_name=__name__)
