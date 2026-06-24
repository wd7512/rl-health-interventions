"""Weather data loader with local caching.

Fetches historical weather from Open-Meteo API and caches responses
as CSV files for offline use.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# US census region centroids (approximate lat/lon) for NHANES regions.
REGION_COORDS: dict[int, tuple[float, float]] = {
    1: (40.7, -74.0),  # Northeast (NYC proxy)
    2: (39.0, -77.0),  # Mid-Atlantic (DC proxy)
    3: (35.2, -80.8),  # Southeast (Charlotte proxy)
    4: (41.9, -87.6),  # Midwest (Chicago proxy)
    5: (32.8, -96.8),  # South Central (Dallas proxy)
    6: (34.1, -118.2),  # West (LA proxy)
    7: (47.6, -122.3),  # Northwest (Seattle proxy)
    8: (39.7, -105.0),  # Mountain (Denver proxy)
}


class WeatherLoader:
    """Loads cached weather data and provides per-day weather lookups.

    Cache format: CSV with columns
        date,region,temp_mean,temp_max,temp_min,precip_sum,wind_speed_10m_max
    """

    def __init__(self, cache_path: str | Path) -> None:
        self._cache_path = Path(cache_path)
        self._cache: dict[tuple[str, int], dict[str, float]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._cache_path.exists():
            logger.warning("Weather cache not found at %s", self._cache_path)
            return
        with open(self._cache_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["date"], int(row["region"]))
                self._cache[key] = {
                    "temp_mean": float(row["temp_mean"]),
                    "temp_max": float(row["temp_max"]),
                    "temp_min": float(row["temp_min"]),
                    "precip_sum": float(row["precip_sum"]),
                    "wind_speed_10m_max": float(row["wind_speed_10m_max"]),
                }
        logger.info(
            "Loaded %d weather records from %s", len(self._cache), self._cache_path
        )

    def get_weather(self, date: str, region: int) -> tuple[float, float]:
        """Get (temperature_c, precipitation_mm) for a date and region.

        Args:
            date: ISO date string (YYYY-MM-DD).
            region: NHANES census region code (1-8).

        Returns:
            (temperature_c, precipitation_mm). Defaults to (15.0, 0.0) if not found.
        """
        self._ensure_loaded()
        key = (date, region)
        if key in self._cache:
            row = self._cache[key]
            return (row["temp_mean"], row["precip_sum"])
        return (15.0, 0.0)

    def get_weather_features(self, date: str, region: int) -> tuple[float, float]:
        """Get (temp_norm, precip_flag) for use in feature construction.

        Returns:
            (temp_norm, precip_flag) as floats in [0, 1].
        """
        temp_c, precip_mm = self.get_weather(date, region)
        temp_norm = float(max(0.0, min(1.0, (temp_c + 10.0) / 50.0)))
        precip_flag = float(min(precip_mm / 10.0, 1.0))
        return (temp_norm, precip_flag)


def register() -> None:
    from rl_health_interventions.data import REGISTRY

    REGISTRY["weather"] = WeatherLoader
