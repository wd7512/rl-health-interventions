"""HeartSteps transition model.

Handles availability simulation based on time-of-day and participant
availability probability.
"""

from __future__ import annotations

import logging

import numpy as np

from rl_health_interventions.transitions._base import TransitionModel

logger = logging.getLogger(__name__)

_NIGHT_WINDOWS = {0, 9}


class HeartStepsTransition(TransitionModel):
    """HeartSteps transition model.

    Returns "available" or "unavailable" based on:
    - Night windows (0, 9) are always unavailable.
    - Day windows are available with probability p_avail.
    """

    def __init__(
        self,
        config: object | None = None,
        p_avail: float = 0.85,
        n_windows: int = 10,
        step_data: np.ndarray | None = None,
        seed: int = 42,
        **_kwargs: object,
    ) -> None:
        self._p_avail = p_avail
        self._n_windows = n_windows
        self._rng = np.random.default_rng(seed)
        self._step_counter = 0

    def transition(self, state: str, action: str) -> str:
        window = self._step_counter % self._n_windows
        self._step_counter += 1
        if window in _NIGHT_WINDOWS:
            return "unavailable"
        return (
            "available" if self._rng.binomial(1, self._p_avail) == 1 else "unavailable"
        )


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY["heartsteps"] = HeartStepsTransition
