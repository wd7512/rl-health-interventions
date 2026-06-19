from __future__ import annotations

from abc import ABC, abstractmethod

from rl_health_interventions.state import StateView


class RewardHandler(ABC):
    """Base class for reward computation.

    Subclasses must implement ``reward()`` to compute the per-step reward
    and ``reset()`` to clear any episode-local state (e.g. accumulators)
    when a new episode begins.
    """

    @abstractmethod
    def reward(
        self, state: StateView, action: str, step_idx: int
    ) -> tuple[float, bool]:
        """Return ``(reward, done)`` for the given transition.

        ``state`` is the post-transition state at the current step.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear episode-local state. Called by ``Environment.reset()``."""
