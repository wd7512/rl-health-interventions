"""Download historical weather data from Open-Meteo for NHANES participants.

Usage:
    uv run python scripts/download_weather.py --participants-csv data/nhanes/participants.csv --output data/weather/

Requires: participants.csv with columns seqn,date,region (from download_nhanes.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REGION_COORDS = {
    1: (40.7, -74.0),
    2: (39.0, -77.0),
    3: (35.2, -80.8),
    4: (41.9, -87.6),
    5: (32.8, -96.8),
    6: (34.1, -118.2),
    7: (47.6, -122.3),
    8: (39.7, -105.0),
}


def fetch_weather_batch(
    dates: list[str], lat: float, lon: float
) -> list[dict[str, float]]:
    """Fetch daily weather for a list of dates from Open-Meteo."""
    if not dates:
        return []
    start_date = min(dates)
    end_date = max(dates)
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
        f"precipitation_sum,wind_speed_10m_max"
        f"&timezone=America/New_York"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.error("Open-Meteo request failed: %s", e)
        return []
    daily = data.get("daily", {})
    result_dates = daily.get("time", [])
    result = []
    for i, d in enumerate(result_dates):
        result.append(
            {
                "date": d,
                "temp_mean": daily.get("temperature_2m_mean", [0.0])[i] or 0.0,
                "temp_max": daily.get("temperature_2m_max", [0.0])[i] or 0.0,
                "temp_min": daily.get("temperature_2m_min", [0.0])[i] or 0.0,
                "precip_sum": daily.get("precipitation_sum", [0.0])[i] or 0.0,
                "wind_speed_10m_max": daily.get("wind_speed_10m_max", [0.0])[i] or 0.0,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download weather data for NHANES participants"
    )
    parser.add_argument(
        "--participants-csv", required=True, help="Path to participants.csv"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for weather cache CSV"
    )
    parser.add_argument(
        "--batch-days", type=int, default=365, help="Days per API request"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / "weather_cache.csv"

    existing: dict[tuple[str, int], dict[str, str | float]] = {}
    if cache_file.exists():
        with open(cache_file) as f:
            for row in csv.DictReader(f):
                existing[(row["date"], int(row["region"]))] = row
        logger.info("Loaded %d existing cache entries", len(existing))

    region_dates: dict[int, set[str]] = {}
    with open(args.participants_csv) as f:
        for row in csv.DictReader(f):
            region = int(row["region"])
            date = row["date"]
            if (date, region) not in existing:
                region_dates.setdefault(region, set()).add(date)

    if not region_dates:
        logger.info("All weather data already cached. Nothing to fetch.")
        return

    all_rows: list[dict] = []
    for region, dates in region_dates.items():
        if region not in REGION_COORDS:
            logger.warning("Unknown region %d, skipping", region)
            continue
        lat, lon = REGION_COORDS[region]
        sorted_dates = sorted(dates)
        for i in range(0, len(sorted_dates), args.batch_days):
            batch = sorted_dates[i : i + args.batch_days]
            logger.info(
                "Fetching weather: region=%d, dates=%s..%s (%d days)",
                region,
                batch[0],
                batch[-1],
                len(batch),
            )
            records = fetch_weather_batch(batch, lat, lon)
            for rec in records:
                rec["region"] = region
                all_rows.append(rec)
            time.sleep(0.5)

    fieldnames = [
        "date",
        "region",
        "temp_mean",
        "temp_max",
        "temp_min",
        "precip_sum",
        "wind_speed_10m_max",
    ]
    with open(cache_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if cache_file.stat().st_size == 0:
            writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    logger.info("Wrote %d new weather records to %s", len(all_rows), cache_file)


if __name__ == "__main__":
    main()
