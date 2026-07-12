from __future__ import annotations


class DosageTracker:
    """Tracks suggestion dosage: X_{t+1} = lambda * X_t + 1_{E_t}.

    E_t is whether any suggestion was sent at the previous decision time.
    """

    def __init__(self, lambda_decay: float = 0.95) -> None:
        if not (0 < lambda_decay < 1):
            msg = "lambda_decay must be in (0, 1)"
            raise ValueError(msg)
        self._lambda = lambda_decay
        self._dosage: float = 0.0

    def update(self, suggestion_sent: bool) -> float:
        """Update dosage and return the new value."""
        self._dosage = self._lambda * self._dosage + (1.0 if suggestion_sent else 0.0)
        return self._dosage

    def get_dosage(self) -> float:
        return self._dosage

    def reset(self) -> None:
        self._dosage = 0.0
