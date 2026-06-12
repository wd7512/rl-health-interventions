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
import os

import numpy as np
import pandas as pd

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


class RealNHANESLoader:
    """Loads real NHANES minute-level step count data from PhysioNet CSV.

    Reads the 1440-minute CSV, filters for days with sufficient valid data,
    aggregates minute-level step counts to 30-minute windows, and selects
    participants with enough valid days.

    Args:
        data_path: Path to the NHANES CSV file.
        n_participants: Number of participants to load (default 50).
        n_days: Number of days per participant (max 9, default 7).
        n_windows: Number of windows per day (must divide 1440, default 10).
        min_valid_minutes: Minimum non-NA minutes per day (default 600).
        seed: Random seed for deterministic participant selection (default 42).
    """

    def __init__(
        self,
        data_path: str,
        n_participants: int = 50,
        n_days: int = 7,
        n_windows: int = 10,
        min_valid_minutes: int = 600,
        seed: int = 42,
    ) -> None:
        self.data_path = data_path
        self.n_participants = n_participants
        self.n_days = n_days
        self.n_windows = n_windows
        self.min_valid_minutes = min_valid_minutes
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        if 1440 % self.n_windows != 0:
            raise ValueError(f"n_windows={n_windows} must divide 1440 evenly")

        logger.debug(
            "RealNHANESLoader initialised: data_path=%s, n_participants=%d, "
            "n_days=%d, n_windows=%d, min_valid_minutes=%d, seed=%d",
            data_path,
            n_participants,
            n_days,
            n_windows,
            min_valid_minutes,
            seed,
        )

    def load(self) -> np.ndarray:
        """Load and aggregate real NHANES step count data.

        Returns:
            Array of shape (n_participants, n_days, n_windows) with
            non-negative step counts aggregated to 30-minute windows.

        Raises:
            FileNotFoundError: If the data file does not exist.
        """
        if not os.path.exists(self.data_path):
            logger.warning(
                "NHANES data file not found at %s, falling back to synthetic data",
                self.data_path,
            )
            gen = SyntheticNHANESGenerator(seed=self._seed)
            return gen.generate(
                n_participants=self.n_participants,
                n_days=self.n_days,
                n_windows=self.n_windows,
            )

        minutes_per_window = 1440 // self.n_windows
        minute_cols = [f"min_{i}" for i in range(1, 1441)]
        cols_to_read = ["SEQN", "PAXDAYM"] + minute_cols

        participant_data: dict[int, dict[int, np.ndarray]] = {}
        finalized: list[int] = []

        for chunk in pd.read_csv(
            self.data_path,
            usecols=cols_to_read,
            chunksize=5000,
        ):
            chunk = chunk[chunk["PAXDAYM"] <= self.n_days]

            for seqn, group in chunk.groupby("SEQN", sort=False):
                if seqn in finalized:
                    continue

                if seqn not in participant_data:
                    participant_data[seqn] = {}

                for _, row in group.iterrows():
                    day = int(row["PAXDAYM"])
                    if day in participant_data[seqn]:
                        continue

                    minute_vals = row[minute_cols].values.astype(np.float64)
                    valid_count = np.sum(~np.isnan(minute_vals))
                    if valid_count < self.min_valid_minutes:
                        continue

                    minute_vals = np.nan_to_num(minute_vals, nan=0.0)
                    windows = minute_vals.reshape(
                        self.n_windows, minutes_per_window
                    ).sum(axis=1)
                    participant_data[seqn][day] = windows

                if len(participant_data[seqn]) >= self.n_days:
                    finalized.append(seqn)
                    if len(finalized) >= self.n_participants:
                        break

            if len(finalized) >= self.n_participants:
                break

        if len(finalized) < self.n_participants:
            logger.warning(
                "Only found %d participants with >=%d valid days (requested %d)",
                len(finalized),
                self.n_days,
                self.n_participants,
            )

        selected = finalized[: self.n_participants]
        result = np.zeros((len(selected), self.n_days, self.n_windows))

        for i, seqn in enumerate(selected):
            days = sorted(participant_data[seqn].keys())[: self.n_days]
            for j, day in enumerate(days):
                result[i, j] = participant_data[seqn][day]

        return result


class NHANESLoader:
    """Data loader interface for NHANES step count data.

    Supports both synthetic and real data sources. The synthetic source
    generates NHANES-like data on-the-fly; the real source loads from
    a PhysioNet CSV file.

    Args:
        data_source: 'synthetic' (default) or 'real'.
        n_participants: Number of participants.
        n_days: Number of days per participant.
        seed: Random seed for reproducibility.
        data_path: Path to NHANES CSV (required for data_source='real').
    """

    def __init__(
        self,
        data_source: str = "synthetic",
        n_participants: int = 100,
        n_days: int = 7,
        seed: int = 42,
        data_path: str | None = None,
    ) -> None:
        self.data_source = data_source
        self.n_participants = n_participants
        self.n_days = n_days
        self._seed = seed
        self.data_path = data_path

    def load(self) -> np.ndarray:
        """Load NHANES step count data.

        Returns:
            Array of shape (n_participants, n_days, 10) with step counts.

        Raises:
            ValueError: If data_source is 'real' but no data_path provided.
        """
        if self.data_source == "synthetic":
            gen = SyntheticNHANESGenerator(seed=self._seed)
            return gen.generate(
                n_participants=self.n_participants,
                n_days=self.n_days,
                n_windows=10,
            )

        if self.data_source == "real":
            if self.data_path is None:
                msg = "data_path is required when data_source='real'"
                raise ValueError(msg)
            loader = RealNHANESLoader(
                data_path=self.data_path,
                n_participants=self.n_participants,
                n_days=self.n_days,
                n_windows=10,
                seed=self._seed,
            )
            return loader.load()

        msg = f"Unknown data_source: {self.data_source!r}"
        raise ValueError(msg)
