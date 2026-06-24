"""NHANES step count data loaders.

Synthetic generator producing NHANES-like step count patterns, and
a real-data loader for PhysioNet minute-level CSV files.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SyntheticNHANESGenerator:
    """Generates synthetic step count data with NHANES-like properties."""

    def __init__(
        self,
        seed: int = 42,
        daily_mean: float = 5000.0,
        daily_std: float = 2000.0,
        tau: float = 0.6,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.daily_mean = daily_mean
        self.daily_std = daily_std
        self.tau = tau
        self._seed = seed
        self._time_weights = np.array(
            [0.3, 0.5, 1.5, 1.8, 1.2, 0.8, 1.0, 1.4, 1.2, 0.4]
        )
        self._time_weights = self._time_weights / self._time_weights.mean()

    def generate(
        self,
        n_participants: int = 100,
        n_days: int = 7,
        n_windows: int = 10,
    ) -> np.ndarray:
        data = np.zeros((n_participants, n_days, n_windows))
        for p in range(n_participants):
            part_mean = self._rng.gamma(
                self.daily_mean / self.daily_std, self.daily_std
            )
            part_mean = max(part_mean, 500.0)
            window_base = (part_mean / n_windows) * self._time_weights[:n_windows]
            prev_resid = 0.0
            for d in range(n_days):
                noise = self._rng.normal(0, 1.0)
                day_scale = self.tau * prev_resid + np.sqrt(1 - self.tau**2) * noise
                day_scale = np.clip(day_scale, -2.0, 2.0)
                prev_resid = day_scale
                for w in range(n_windows):
                    base = max(window_base[w] * (1.0 + day_scale * 0.3), 0.0)
                    count = int(self._rng.poisson(base))
                    data[p, d, w] = max(count, 0)
        return data


class RealNHANESLoader:
    """Loads real NHANES minute-level step count data from PhysioNet CSV."""

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

    def load(
        self,
    ) -> tuple[np.ndarray, list[dict]]:
        """Load step data and participant metadata.

        Returns:
            (step_data, participant_meta) where step_data is
            (n_participants, n_days, n_windows) and participant_meta is a
            list of dicts with 'seqn', 'date', 'region' keys.
        """
        import os

        import pandas as pd

        if not os.path.exists(self.data_path):
            logger.warning(
                "NHANES data not found at %s, falling back to synthetic",
                self.data_path,
            )
            gen = SyntheticNHANESGenerator(seed=self._seed)
            data = gen.generate(
                n_participants=self.n_participants,
                n_days=self.n_days,
                n_windows=self.n_windows,
            )
            meta = [
                {"seqn": i, "date": "2004-01-01", "region": 1}
                for i in range(self.n_participants)
            ]
            return data, meta

        minutes_per_window = 1440 // self.n_windows
        minute_cols = [f"min_{i}" for i in range(1, 1441)]
        cols_to_read = ["SEQN", "PAXDAYM"] + minute_cols
        participant_data: dict[int, dict[int, np.ndarray]] = {}
        finalized: list[int] = []

        for chunk in pd.read_csv(self.data_path, usecols=cols_to_read, chunksize=5000):
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

        selected = finalized[: self.n_participants]
        result = np.zeros((len(selected), self.n_days, self.n_windows))
        meta_list: list[dict] = []
        for i, seqn in enumerate(selected):
            days = sorted(participant_data[seqn].keys())[: self.n_days]
            for j, day in enumerate(days):
                result[i, j] = participant_data[seqn][day]
            meta_list.append({"seqn": int(seqn), "date": "2004-01-01", "region": 1})

        logger.info(
            "Loaded %d participants x %d days x %d windows from %s",
            len(selected),
            self.n_days,
            self.n_windows,
            self.data_path,
        )
        return result, meta_list

    def load_with_participants_csv(
        self, participants_csv: str
    ) -> tuple[np.ndarray, list[dict]]:
        """Load step data and merge with participants.csv metadata."""
        import os

        import pandas as pd

        step_data, meta = self.load()

        if os.path.exists(participants_csv):
            pdf = pd.read_csv(participants_csv)
            meta_map = {
                int(row["seqn"]): {"date": row["date"], "region": int(row["region"])}
                for _, row in pdf.iterrows()
            }
            for m in meta:
                if m["seqn"] in meta_map:
                    m["date"] = meta_map[m["seqn"]]["date"]
                    m["region"] = meta_map[m["seqn"]]["region"]
            logger.info("Merged participant metadata from %s", participants_csv)

        return step_data, meta


class NHANESLoader:
    """Facade: dispatches to synthetic or real NHANES data."""

    def __init__(
        self,
        data_source: str = "synthetic",
        n_participants: int = 100,
        n_days: int = 42,
        seed: int = 42,
        data_path: str | None = None,
    ) -> None:
        self.data_source = data_source
        self.n_participants = n_participants
        self.n_days = n_days
        self._seed = seed
        self.data_path = data_path

    def load(self) -> tuple[np.ndarray, list[dict]]:
        if self.data_source == "synthetic":
            gen = SyntheticNHANESGenerator(seed=self._seed)
            data = gen.generate(
                n_participants=self.n_participants,
                n_days=self.n_days,
                n_windows=10,
            )
            meta = [
                {"seqn": i, "date": "2004-01-01", "region": 1}
                for i in range(self.n_participants)
            ]
            return data, meta
        if self.data_source == "real":
            if self.data_path is None:
                msg = "data_path is required when data_source='real'"
                raise ValueError(msg)
            loader = RealNHANESLoader(
                data_path=self.data_path,
                n_participants=self.n_participants,
                n_days=self.n_days,
                seed=self._seed,
            )
            return loader.load()
        msg = f"Unknown data_source: {self.data_source!r}"
        raise ValueError(msg)


def register() -> None:
    from rl_health_interventions.data import REGISTRY

    REGISTRY["nhanes_synthetic"] = SyntheticNHANESGenerator
    REGISTRY["nhanes_real"] = RealNHANESLoader
    REGISTRY["nhanes"] = NHANESLoader
