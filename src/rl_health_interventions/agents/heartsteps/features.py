"""HeartSteps feature construction.

g(s): 6 features — current steps, daily total, sin/cos time-of-day,
       day-of-week, step variability.
f(s): 4 features — current steps, step variability, time-of-day sin,
       daily total.
"""

from __future__ import annotations

import numpy as np


def construct_heartsteps_features(
    step_data: np.ndarray,
    t: int,
    daily_steps: float,
    time_slot: int,
    day: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct baseline g(s) and treatment f(s) feature vectors.

    Args:
        step_data: Flattened 1-D step counts for the participant.
        t: Current time index into step_data.
        daily_steps: Running daily step total.
        time_slot: Decision time slot (0-4 for 5 slots/day with 10 windows).
        day: Study day index.

    Returns:
        Tuple of (g_features, f_features) as numpy arrays.
    """
    steps_30min = float(step_data[t])
    lookback_var = max(0, t - 5)
    n_var = t - lookback_var + 1
    step_var = float(np.std(step_data[lookback_var : t + 1])) if n_var >= 2 else 0.0

    steps_30min_norm = min(steps_30min / 500.0, 1.0)
    daily_norm = min(daily_steps / 10000.0, 1.0)
    time_of_day_sin = np.sin(2.0 * np.pi * time_slot / 5.0)
    time_of_day_cos = np.cos(2.0 * np.pi * time_slot / 5.0)
    day_of_week_sin = np.sin(2.0 * np.pi * (day % 7) / 7.0)
    step_var_norm = min(step_var / 100.0, 1.0)

    g = np.array(
        [
            steps_30min_norm,
            daily_norm,
            time_of_day_sin,
            time_of_day_cos,
            day_of_week_sin,
            step_var_norm,
        ]
    )

    f = np.array(
        [
            steps_30min_norm,
            step_var_norm,
            time_of_day_sin,
            daily_norm,
        ]
    )

    return g, f
