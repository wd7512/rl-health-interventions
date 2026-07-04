from __future__ import annotations

import numpy as np

from rl_health_interventions.agents._base import Agent


class ContextualBanditAgent(Agent):
    """Base for bandit agents with optional contextual action selection.

    Subclasses implement ``select_action`` and ``update`` using
    ``_get_context_key`` to route per-action parameters based on the
    current state context.
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        seed: int = 42,
        contextual: bool = False,
        context_feature: str | list[str] | None = None,
    ) -> None:
        if contextual and (
            context_feature is None
            or (isinstance(context_feature, str) and not context_feature.strip())
            or (isinstance(context_feature, list) and not context_feature)
            or not isinstance(context_feature, (str, list))
        ):
            raise ValueError(
                "context_feature must be a non-empty string or list when contextual=True"
            )
        self.contextual = contextual
        self.context_feature = context_feature
        self._actions = actions or ["nudge", "idle"]
        self._rng = np.random.default_rng(seed)

    def _get_context_key(self, state, action: str) -> str | tuple[str, ...]:
        """Return a key for parameter lookup.

        Non-contextual mode: just the action name.
        Contextual mode with string feature: ``(context_value, action)`` tuple.
        Contextual mode with list of features: ``(val1, val2, ..., action)`` tuple.
        """
        if not self.contextual:
            return action
        if state is None:
            raise ValueError("state cannot be None when contextual=True")
        assert self.context_feature is not None
        if isinstance(self.context_feature, str):
            value = getattr(state, self.context_feature)
            return (value, action)
        values = tuple(getattr(state, f) for f in self.context_feature)
        return values + (action,)
