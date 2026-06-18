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
        context_feature: str | None = None,
    ) -> None:
        self.contextual = contextual
        self.context_feature = context_feature
        self._actions = actions or ["nudge", "idle"]
        self._rng = np.random.default_rng(seed)

    def _get_context_key(self, state, action: str) -> str | tuple[str, str]:
        """Return a key for parameter lookup.

        Non-contextual mode: just the action name.
        Contextual mode: ``(context_value, action)`` tuple.
        """
        if not self.contextual:
            return action
        if self.context_feature is None:
            raise ValueError("context_feature must be set when contextual=True")
        value = getattr(state, self.context_feature, None)
        if value is None:
            raise ValueError(
                f"State has no attribute '{self.context_feature}'. "
                f"Available attributes: "
                f"{[a for a in dir(state) if not a.startswith('_')]}"
            )
        return (value, action)
