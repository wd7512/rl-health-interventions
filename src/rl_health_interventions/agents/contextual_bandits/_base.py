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
        if contextual and context_feature is None:
            raise ValueError("context_feature must be provided when contextual=True")
        if context_feature is not None and not isinstance(context_feature, (str, list)):
            raise ValueError("context_feature must be a string or list of strings")
        if isinstance(context_feature, str) and not context_feature.strip():
            raise ValueError(
                "context_feature must be a non-empty string when contextual=True"
            )
        if isinstance(context_feature, list) and (
            not context_feature
            or any(not isinstance(f, str) or not f.strip() for f in context_feature)
        ):
            raise ValueError(
                "context_feature must be a non-empty list of non-empty strings "
                "when contextual=True"
            )
        self.contextual = contextual
        self._context_features: list[str] | None = (
            [context_feature] if isinstance(context_feature, str) else context_feature
        )
        self._actions = actions or ["nudge", "idle"]
        self._rng = np.random.default_rng(seed)

    def _get_context_key(self, state, action: str) -> str | tuple:
        """Return a key for parameter lookup.

        Non-contextual mode: just the action name.
        Contextual mode with single feature: ``(context_value, action)`` tuple.
        Contextual mode with multiple features: ``((v1, v2, ...), action)`` tuple.
        """
        if not self.contextual:
            return action
        if state is None:
            raise ValueError("state cannot be None when contextual=True")
        assert self._context_features is not None
        if len(self._context_features) == 1:
            value = getattr(state, self._context_features[0])
            return (value, action)
        values = tuple(getattr(state, f) for f in self._context_features)
        return (values, action)
