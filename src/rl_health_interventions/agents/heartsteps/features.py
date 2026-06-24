"""HeartSteps feature construction.

g(s): 12 features — current steps, daily total, sin/cos time-of-day,
       sin/cos day-of-week, weekend flag, step variability, 1hr avg,
       24hr avg, temperature, precipitation.
f(s): 8 features — current steps, step variability, time-of-day sin,
       daily total, weekend flag, temperature, precipitation, 1hr avg.
"""

from __future__ import annotations

import numpy as np


def construct_heartsteps_features(
    step_data: np.ndarray,
    t: int,
    daily_steps: float,
    time_slot: int,
    day: int,
    weather: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct baseline g(s) and treatment f(s) feature vectors.

    Args:
        step_data: Flattened 1-D step counts for the participant.
        t: Current time index into step_data.
        daily_steps: Running daily step total.
        time_slot: Decision time slot (0-4 for 5 slots/day with 10 windows).
        day: Study day index.
        weather: Optional (temperature_c, precipitation_mm) tuple.

    Returns:
        Tuple of (g_features, f_features) as numpy arrays.
    """
    steps_30min = float(step_data[t])
    lookback_var = max(0, t - 5)
    n_var = t - lookback_var + 1
    step_var = float(np.std(step_data[lookback_var : t + 1])) if n_var >= 2 else 0.0

    lookback_1hr = max(0, t - 1)
    steps_1hr = float(np.mean(step_data[lookback_1hr : t + 1]))
    lookback_24hr = max(0, t - 47)
    steps_24hr = float(np.mean(step_data[lookback_24hr : t + 1]))

    steps_30min_norm = min(steps_30min / 500.0, 1.0)
    daily_norm = min(daily_steps / 10000.0, 1.0)
    time_of_day_sin = np.sin(2.0 * np.pi * time_slot / 5.0)
    time_of_day_cos = np.cos(2.0 * np.pi * time_slot / 5.0)
    day_of_week_sin = np.sin(2.0 * np.pi * (day % 7) / 7.0)
    day_of_week_cos = np.cos(2.0 * np.pi * (day % 7) / 7.0)
    is_weekend = float((day % 7) >= 5)
    step_var_norm = min(step_var / 100.0, 1.0)
    steps_1hr_norm = min(steps_1hr / 500.0, 1.0)
    steps_24hr_norm = min(steps_24hr / 1000.0, 1.0)

    if weather is not None:
        temp_norm = float(np.clip((weather[0] + 10.0) / 50.0, 0.0, 1.0))
        precip_flag = float(min(weather[1] / 10.0, 1.0))
    else:
        temp_norm = 0.5
        precip_flag = 0.0

    g = np.array(
        [
            steps_30min_norm,
            daily_norm,
            time_of_day_sin,
            time_of_day_cos,
            day_of_week_sin,
            day_of_week_cos,
            is_weekend,
            step_var_norm,
            steps_1hr_norm,
            steps_24hr_norm,
            temp_norm,
            precip_flag,
        ]
    )

    f = np.array(
        [
            steps_30min_norm,
            step_var_norm,
            time_of_day_sin,
            daily_norm,
            is_weekend,
            temp_norm,
            precip_flag,
            steps_1hr_norm,
        ]
    )

    return g, f
