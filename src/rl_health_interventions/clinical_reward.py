"""Synthetic clinical outcome simulator and reward computation for 3-week signals."""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ClinicalParams:
    """Parameters for the synthetic clinical outcome model."""
    baseline_blood_pressure: float = 130.0
    baseline_hba1c: float = 7.0
    step_benefit_scale: float = -0.5  # per 1000 steps above threshold
    threshold_steps: float = 7000.0   # steps per day threshold
    noise_std: float = 2.0
    improvement_max: float = 10.0     # max reduction in bp


class ClinicalOutcomeSimulator:
    """Simulates 3-week clinical outcomes based on daily step counts.

    Models blood pressure and HbA1c as functions of cumulative step performance
    over a 3-week window, with realistic noise and saturation effects.
    """

    def __init__(self, params: Optional[ClinicalParams] = None):
        self.params = params or ClinicalParams()
        self._step_buffer: list[float] = []

    def reset(self) -> None:
        """Clear accumulated step data."""
        self._step_buffer = []

    def add_daily_steps(self, steps: float) -> None:
        """Record daily step count (called each day).

        Args:
            steps: Total steps for the day.
        """
        self._step_buffer.append(steps)
        # Keep only last 21 days (3 weeks)
        if len(self._step_buffer) > 21:
            self._step_buffer = self._step_buffer[-21:]

    def compute_clinical_change(self) -> dict[str, float]:
        """Compute simulated change in clinical metrics after 3 weeks.

        Returns:
            Dictionary with 'blood_pressure_change' and 'hba1c_change'.
        """
        if len(self._step_buffer) < 21:
            # Not enough data; return no change
            return {'blood_pressure_change': 0.0, 'hba1c_change': 0.0}

        avg_steps = np.mean(self._step_buffer)
        excess_steps = max(0, avg_steps - self.params.threshold_steps)
        # Benefit per 1000 steps above threshold
        benefit = self.params.step_benefit_scale * (excess_steps / 1000.0)
        # Clip to maximum improvement
        bp_change = max(benefit, -self.params.improvement_max)
        hba1c_change = bp_change * 0.2  # correlate HbA1c with BP (simplified)
        # Add noise
        bp_change += np.random.normal(0, self.params.noise_std)
        hba1c_change += np.random.normal(0, self.params.noise_std * 0.3)
        return {
            'blood_pressure_change': bp_change,
            'hba1c_change': hba1c_change
        }

    def __len__(self) -> int:
        return len(self._step_buffer)


def compute_clinical_reward(
    simulator: ClinicalOutcomeSimulator,
    bp_scale: float = -0.5,
    hba1c_scale: float = -1.0
) -> float:
    """Compute reward contribution from clinical outcome.

    Args:
        simulator: Clinical simulator with accumulated data.
        bp_scale: Weight for blood pressure change (negative because decrease is good).
        hba1c_scale: Weight for HbA1c change.

    Returns:
        Reward value (negative for worsening, positive for improvement).
    """
    changes = simulator.compute_clinical_change()
    reward = (
        bp_scale * changes['blood_pressure_change'] +
        hba1c_scale * changes['hba1c_change']
    )
    return reward
