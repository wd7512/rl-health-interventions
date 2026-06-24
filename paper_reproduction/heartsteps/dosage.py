"""Dosage variable tracking cumulative treatment exposure.

Implements the dosage variable from Section 5.2 of the HeartSteps V2 paper:

    X_{t+1} = lambda * X_t + 1{treatment or anti-sedentary suggestion}

where lambda is the decay rate (default 0.95). The dosage variable captures
the user's recent intervention history and is used to form the proxy for
delayed effects on future rewards.

Reference:
    Liao et al. (2019). "Personalized HeartSteps." arXiv:1909.03539, Section 5.2.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class DosageTracker:
    """Tracks cumulative treatment exposure via exponential moving average.

    The dosage variable X quantifies how much treatment the user has received
    recently. It decays exponentially (by factor lambda each step) and increases
    by 1 whenever a treatment event occurs (activity suggestion or
    anti-sedentary message).

    Attributes:
        value: Current dosage value.
        decay: Exponential decay rate (lambda in the paper).
    """

    def __init__(self, decay: float = 0.95, initial_value: float = 0.0) -> None:
        """Initialise dosage tracker.

        Args:
            decay: Decay rate lambda. Must be in (0, 1]. Default 0.95 per paper.
            initial_value: Starting dosage. Default 0.0 for fresh participants.
        """
        if not 0 < decay <= 1.0:
            msg = f"decay must be in (0, 1], got {decay}"
            raise ValueError(msg)
        self.decay = decay
        self._value = initial_value
        logger.debug(
            "DosageTracker initialised: decay=%.2f, initial=%.4f", decay, initial_value
        )

    @property
    def value(self) -> float:
        """Current dosage value."""
        return self._value

    def update(
        self,
        treatment_delivered: bool = False,
        anti_sedentary_delivered: bool = False,
    ) -> None:
        """Advance dosage by one step.

        The event indicator E_t = 1 if either a treatment (activity suggestion)
        or an anti-sedentary message was delivered since the last decision time.

        Args:
            treatment_delivered: Whether an activity suggestion was sent at this
                decision time (A_t = 1).
            anti_sedentary_delivered: Whether an anti-sedentary suggestion was
                sent between decision times (external to the RL algorithm).
        """
        event = treatment_delivered or anti_sedentary_delivered
        event_int = 1 if event else 0
        self._value = self.decay * self._value + event_int
        logger.debug(
            "Dosage updated: treatment=%s, anti_sed=%s, event=%d, new_value=%.4f",
            treatment_delivered,
            anti_sedentary_delivered,
            event_int,
            self._value,
        )

    def reset(self, initial_value: float = 0.0) -> None:
        """Reset dosage to initial value (e.g., for a new episode).

        Args:
            initial_value: Value to reset to. Default 0.0.
        """
        self._value = initial_value
        logger.debug("Dosage reset to %.4f", initial_value)
