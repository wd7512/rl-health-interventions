"""Dosage variable tracking cumulative treatment exposure.

X_{t+1} = lambda * X_t + 1{treatment or anti-sedentary suggestion}.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class DosageTracker:
    """Tracks cumulative treatment exposure via exponential moving average."""

    def __init__(self, decay: float = 0.95) -> None:
        if not 0 < decay <= 1.0:
            msg = f"decay must be in (0, 1], got {decay}"
            raise ValueError(msg)
        self.decay = decay
        self._value = 0.0

    @property
    def value(self) -> float:
        return self._value

    def update(
        self,
        treatment_delivered: bool = False,
        anti_sedentary_delivered: bool = False,
    ) -> None:
        event = 1 if (treatment_delivered or anti_sedentary_delivered) else 0
        self._value = self.decay * self._value + event

    def reset(self) -> None:
        self._value = 0.0
