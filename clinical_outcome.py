"""Synthetic 3-week clinical outcome signal for reward function."""
import numpy as np


def simulate_clinical_outcome(steps_history: list) -> float:
    """
    Simulates a clinical outcome score based on the last 21 days of step counts.
    Returns a float in [0, 1] where higher is better.
    Uses a linear model: clinical_improvement = min(1, avg_daily_steps / 10000)
    """
    if len(steps_history) == 0:
        return 0.0
    # Use last 21 days (or fewer if insufficient history)
    window = steps_history[-21:] if len(steps_history) >= 21 else steps_history
    avg_steps = np.mean(window)
    score = min(1.0, avg_steps / 10000.0)
    return score
