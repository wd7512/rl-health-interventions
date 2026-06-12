"""NHANES-style step count data loader.

Provides synthetic step count data with NHANES-like properties (CC0, public domain)
and a loader interface that can switch between synthetic and real data sources.

The synthetic generator produces minute-level step counts aggregated to 30-minute
windows, with realistic time-of-day patterns and between-participant variability.

Reference:
    https://physionet.org/content/minute-level-step-count-nhanes/
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SyntheticNHANESGenerator:
    """Generates synthetic step count data with NHANES-like properties.

    Produces realistic step count patterns with:
    - Morning peak (highest activity around 9-11am)
    - Afternoon decline
    - Evening secondary peak
    - Late-night low activity
    - Between-participant variability
    - Day-to-day autocorrelation
    """

    def __init__(
        self,
        seed: int = 42,
        daily_mean: float = 5000.0,
        daily_std: float = 2000.0,
        tau: float = 0.6,
    ) -> None:
        """Initialise generator.

        Args:
            seed: Random seed for reproducibility.
            daily_mean: Mean daily step count. Default 5000.
            daily_std: Standard deviation of daily steps. Default 2000.
            tau: Day-to-day autocorrelation (0-1). Default 0.6.
        """
        self._rng = np.random.default_rng(seed)
        self.daily_mean = daily_mean
        self.daily_std = daily_std
        self.tau = tau
        self._seed = seed

        # Time-of-day pattern weights for 10 windows (0 = late night, 9 = late evening)
        # Based on typical NHANES wear time patterns
        self._time_weights = np.array(
            [0.3, 0.5, 1.5, 1.8, 1.2, 0.8, 1.0, 1.4, 1.2, 0.4]
        )
        self._time_weights = self._time_weights / self._time_weights.mean()

        logger.debug(
            "SyntheticNHANESGenerator initialised: seed=%d, daily_mean=%.0f, "
            "daily_std=%.0f, tau=%.2f",
            seed,
            daily_mean,
            daily_std,
            tau,
        )

    def generate(
        self,
        n_participants: int = 100,
        n_days: int = 7,
        n_windows: int = 10,
    ) -> np.ndarray:
        """Generate synthetic step count data.

        Args:
            n_participants: Number of participants.
            n_days: Number of days per participant.
            n_windows: Number of 30-minute windows per day (default 10).

        Returns:
            Array of shape (n_participants, n_days, n_windows) with
            non-negative integer step counts.
        """
        data = np.zeros((n_participants, n_days, n_windows))

        for p in range(n_participants):
            # Per-participant mean (variance across participants)
            participant_mean = self._rng.gamma(
                self.daily_mean / self.daily_std, self.daily_std
            )
            participant_mean = max(participant_mean, 500.0)

            # Per-window baseline = participant_mean / n_windows * time_weight
            window_base = (participant_mean / n_windows) * self._time_weights[
                :n_windows
            ]

            # Generate days with autocorrelation
            prev_residual = 0.0
            for d in range(n_days):
                # Autoregressive day-to-day variation
                noise = self._rng.normal(0, 1.0)
                day_scale = self.tau * prev_residual + np.sqrt(1 - self.tau**2) * noise
                day_scale = np.clip(day_scale, -2.0, 2.0)
                prev_residual = day_scale

                # Per-window step counts
                for w in range(n_windows):
                    base = max(window_base[w] * (1.0 + day_scale * 0.3), 0.0)
                    count = int(self._rng.poisson(base))
                    data[p, d, w] = max(count, 0)

        return data


class NHANESLoader:
    """Data loader interface for NHANES step count data.

    Supports both synthetic and real data sources. The synthetic source
    generates NHANES-like data on-the-fly; the real source will download
    from PhysioNet.

    Args:
        data_source: 'synthetic' (default) or 'real'.
        n_participants: Number of participants (synthetic).
        n_days: Number of days (synthetic).
        seed: Random seed (synthetic).
    """

    def __init__(
        self,
        data_source: str = "synthetic",
        n_participants: int = 100,
        n_days: int = 7,
        seed: int = 42,
    ) -> None:
        self.data_source = data_source
        self.n_participants = n_participants
        self.n_days = n_days
        self._seed = seed

    def load(self) -> np.ndarray:
        """Load NHANES step count data.

        Returns:
            Array of shape (n_participants, n_days, 10) with step counts.

        Raises:
            NotImplementedError: If data_source is 'real'.
        """
        if self.data_source == "synthetic":
            gen = SyntheticNHANESGenerator(seed=self._seed)
            return gen.generate(
                n_participants=self.n_participants,
                n_days=self.n_days,
                n_windows=10,
            )
        msg = "real NHANES data not yet implemented — use data_source='synthetic'"
        raise NotImplementedError(msg)
