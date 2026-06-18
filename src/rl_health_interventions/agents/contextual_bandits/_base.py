from __future__ import annotations

from typing import Any

import numpy as np

from rl_health_interventions.agents._base import Agent


class ContextualBanditAgent(Agent):
    """Abstract base for contextual and non-contextual bandit agents.

    Subclasses inherit _get_context_key for routing state+action to a
    dict key, as well as common constructor parameters.
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        seed: int = 42,
        contextual: bool = False,
        context_feature: str | None = None,
    ) -> None:
        self._actions: list[str] = actions or ["nudge", "idle"]
        self._rng = np.random.default_rng(seed)
        self.contextual = contextual
        self.context_feature = context_feature

    def _get_context_key(self, state: Any, action: str) -> Any:
        """Return the dict key to use for this (state, action) pair.

        In non-contextual mode the key is just the action string.
        In contextual mode the key is a (context_value, action) tuple.
        """
        if not self.contextual:
            return action

        if self.context_feature is None:
            raise ValueError(
                "context_feature must be set when contextual=True"
            )

        try:
            ctx_value = getattr(state, self.context_feature)
        except AttributeError:
            available = list(vars(state).keys())
            raise ValueError(
                f"State has no attribute '{self.context_feature}'. "
                f"Available attributes: {available}"
            )

        return (ctx_value, action)