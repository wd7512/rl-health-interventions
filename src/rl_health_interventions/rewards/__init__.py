from __future__ import annotations

from rl_health_interventions._registry import Registry
from rl_health_interventions.rewards._base import RewardHandler
from rl_health_interventions.rewards import expression

REGISTRY: Registry = Registry("reward")


def make(name_or_config=None, **kwargs) -> RewardHandler:
    if name_or_config is not None and not isinstance(name_or_config, str):
        name = "expression"
        kwargs.setdefault("config", name_or_config)
    elif isinstance(name_or_config, str):
        name = name_or_config
    elif "name" in kwargs:
        name = kwargs.pop("name")
    else:
        raise TypeError("make() requires either a config or name argument")
    return REGISTRY.make(name, **kwargs)


_REWARD_MODULES = [expression]

REGISTRY.load_modules(_REWARD_MODULES, logger_name=__name__)
