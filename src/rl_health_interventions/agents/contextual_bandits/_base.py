from __future__ import annotations

import numpy as np
from typing import Any

from rl_health_interventions.agents._base import Agent


class _NumpyDictWrapper:
    """Wrapper to provide dict-like access to NumPy arrays for backward compatibility."""

    def __init__(self, array: np.ndarray, action_to_index: dict[str, int]) -> None:
        self.array = array
        self.action_to_index = action_to_index

    def __getitem__(self, key: str) -> Any:
        if isinstance(key, str):
            return self.array[self.action_to_index[key]]
        return self.array[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(key, str):
            self.array[self.action_to_index[key]] = value
        else:
            self.array[key] = value

    def __contains__(self, key: str) -> bool:
        if isinstance(key, str):
            return key in self.action_to_index
        return key < len(self.array)

    def __repr__(self) -> str:
        return repr(dict(self.items()))

    def items(self) -> list[tuple[str, Any]]:
        return [(action, self.array[idx]) for action, idx in self.action_to_index.items()]

    def keys(self) -> list[str]:
        return list(self.action_to_index.keys())

    def values(self) -> list[Any]:
        return [self.array[self.action_to_index[action]] for action in self.action_to_index]

    def get(self, key: str, default: Any = None) -> Any:
        if isinstance(key, str) and key in self.action_to_index:
            return self.array[self.action_to_index[key]]
        return default


class ContextualBanditAgent(Agent):
    """Base for bandit agents with optional contextual action selection.

    Subclasses implement ``select_action`` and ``update`` using
    ``_get_context_key`` to route per-action parameters based on the
    current state context.

    For non-contextual agents, uses NumPy arrays internally for Q-values and counts
    for better performance, but provides dict-like access for backward compatibility.
    For contextual agents, uses dicts with tuple keys.
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        seed: int = 42,
        contextual: bool = False,
        context_features: str | list[str] | None = None,
    ) -> None:
        if contextual and (
            context_features is None
            or (isinstance(context_features, str) and not context_features.strip())
            or (
                isinstance(context_features, list)
                and (
                    not context_features
                    or not all(
                        isinstance(f, str) and f.strip() for f in context_features
                    )
                )
            )
            or not isinstance(context_features, (str, list))
        ):
            raise ValueError(
                "context_features must be a non-empty string or list"
                " of non-empty strings when contextual=True"
            )
        self.contextual = contextual
        self.context_features = (
            tuple(context_features)
            if isinstance(context_features, list)
            else context_features
        )
        self._actions = actions or ["nudge", "idle"]
        self._n_actions = len(self._actions)
        self._rng = np.random.default_rng(seed)
        # For non-contextual mode, pre-allocate NumPy arrays
        # For contextual mode, use dicts (keys are dynamic tuples)
        self._use_numpy = not contextual
        # Create action to index mapping for non-contextual mode
        self._action_to_index = {action: i for i, action in enumerate(self._actions)}

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
        assert self.context_features is not None
        if isinstance(self.context_features, str):
            value = getattr(state, self.context_features)
            return (value, action)
        values = tuple(getattr(state, f) for f in self.context_features)
        return (*values, action)
