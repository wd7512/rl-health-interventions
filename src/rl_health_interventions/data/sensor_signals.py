"""Synthetic wearable sensor signal generator for NormWear POC.

Generates realistic-looking PPG and accelerometer signals from discrete
state factors. These signals are fed through NormWear to produce real
FM embeddings.

In production, this would be replaced by actual sensor data from a
wearable device. The key insight: even synthetic signals that encode
the same information as discrete factors will produce embeddings that
capture waveform-level patterns (heart rate variability, movement
intensity, etc.) that discrete factors miss.
"""

from __future__ import annotations

import numpy as np

# Sampling rate matching NormWear's pretraining (64 Hz)
SAMPLING_RATE = 64

# Signal length: 256 samples = 4 seconds at 64 Hz
# NormWear expects [batch, channels, sequence_length]
SEQUENCE_LENGTH = 256

# Number of channels: PPG + accel_x + accel_y (3 channels)
N_CHANNELS = 3


def _generate_ppg(
    step_bin: str,
    sleep: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a synthetic PPG signal based on state.

    PPG characteristics vary with activity level and sleep quality:
    - inactive: low amplitude, steady rhythm (~60 bpm)
    - moderate: medium amplitude, moderate rhythm (~80 bpm)
    - active: high amplitude, fast rhythm (~100 bpm)
    - poor sleep: irregular rhythm, lower amplitude
    """
    # Base heart rate from step_bin
    hr_map = {"inactive": 60, "moderate": 80, "active": 100}
    hr = hr_map.get(step_bin, 70)

    # Poor sleep adds irregularity
    if sleep == "poor":
        hr += rng.normal(0, 5)
        irregularity = 0.3
    else:
        irregularity = 0.1

    t = np.linspace(0, SEQUENCE_LENGTH / SAMPLING_RATE, SEQUENCE_LENGTH)

    # PPG: sinusoidal with harmonics (mimics arterial pulse)
    freq = hr / 60.0  # Hz
    signal = (
        0.6 * np.sin(2 * np.pi * freq * t)
        + 0.25 * np.sin(2 * np.pi * 2 * freq * t + 0.5)
        + 0.1 * np.sin(2 * np.pi * 3 * freq * t + 1.0)
    )

    # Add noise
    signal += rng.normal(0, irregularity, size=SEQUENCE_LENGTH)

    # Normalise to [0, 1]
    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    return signal.astype(np.float32)


def _generate_accel(
    step_bin: str,
    day_of_week: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic accelerometer X and Y signals.

    Movement intensity varies with activity and time of day.
    Weekend patterns differ from weekday.
    """
    # Movement intensity from step_bin
    intensity_map = {"inactive": 0.05, "moderate": 0.3, "active": 0.8}
    intensity = intensity_map.get(step_bin, 0.2)

    # Weekend has more movement variation
    if day_of_week == "weekend":
        intensity *= 1.2

    t = np.linspace(0, SEQUENCE_LENGTH / SAMPLING_RATE, SEQUENCE_LENGTH)

    # Accelerometer: movement patterns + gravity
    accel_x = intensity * np.sin(
        2 * np.pi * 2.0 * t + rng.uniform(0, 2 * np.pi)
    ) + rng.normal(0, 0.1, size=SEQUENCE_LENGTH)
    accel_y = intensity * np.cos(
        2 * np.pi * 1.5 * t + rng.uniform(0, 2 * np.pi)
    ) + rng.normal(0, 0.1, size=SEQUENCE_LENGTH)

    # Normalise to [-1, 1]
    for arr in [accel_x, accel_y]:
        max_abs = max(np.abs(arr).max(), 1e-8)
        arr /= max_abs

    return accel_x.astype(np.float32), accel_y.astype(np.float32)


def generate_sensor_signals(
    state_factors: dict[str, str],
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic wearable sensor signals from state factors.

    Returns:
        Tensor of shape [1, 3, 256] ready for NormWear inference.
        Channels: [PPG, accel_x, accel_y]
    """
    rng = np.random.default_rng(seed)

    ppg = _generate_ppg(
        step_bin=state_factors.get("step_bin", "moderate"),
        sleep=state_factors.get("sleep", "good"),
        rng=rng,
    )
    accel_x, accel_y = _generate_accel(
        step_bin=state_factors.get("step_bin", "moderate"),
        day_of_week=state_factors.get("day_of_week", "weekday"),
        rng=rng,
    )

    # Stack into [channels, sequence_length] then add batch dim
    signals = np.stack([ppg, accel_x, accel_y], axis=0)  # [3, 256]
    return signals[np.newaxis, :, :]  # [1, 3, 256]
