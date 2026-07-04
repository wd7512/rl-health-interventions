from __future__ import annotations

import logging
from typing import Type

from rl_health_interventions.rewards._base import RewardHandler
from rl_health_interventions.rewards import expression

logger = logging.getLogger(__name__)

REGISTRY: dict[str, Type[RewardHandler]] = {}


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
    if name not in REGISTRY:
        raise KeyError(f"Unknown reward handler: {name}. Known: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)


_REWARD_MODULES = [expression]

for _mod in _REWARD_MODULES:
    try:
        _mod.register()
    except Exception:
        logger.exception("Failed to register %s", _mod.__name__)
