"""Parsing for PEARL LLM bootstrap responses.

Parses 7-day step history responses and bins raw step counts into
state factors for the 12-action PEARL config.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

# Binning thresholds (calibrated to PEARL Table 3)
_STEPS_LOW_UPPER = 4000
_STEPS_HIGH_LOWER = 7000
_WALK_PATTERN_HIGH_THRESHOLD = 5000
_MORNING_RATIO_LOW = 0.4
_MORNING_RATIO_HIGH = 0.6


def parse_day_history(response: str) -> list[dict[str, int]] | None:  # noqa: C901, PLR0912
    """Parse a 7-day step history from LLM response.

    Expected format: 7 JSON objects, each with day, morning_steps, afternoon_steps.
    The LLM may output these on separate lines or in a single block.

    Returns list of 7 dicts with keys: day, morning_steps, afternoon_steps.
    Returns None on parse failure.
    """
    results = []
    lines = response.strip().split("\n")

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not isinstance(obj, dict):
            continue

        day = obj.get("day")
        morning = obj.get("morning_steps")
        afternoon = obj.get("afternoon_steps")

        if day is None or morning is None or afternoon is None:
            continue

        if not isinstance(morning, (int, float)) or not isinstance(
            afternoon, (int, float)
        ):
            continue

        if morning < 0 or afternoon < 0:
            continue

        results.append(
            {
                "day": int(day),
                "morning_steps": int(morning),
                "afternoon_steps": int(afternoon),
            }
        )

    if len(results) == 0:
        logger.warning("No valid day records found in response")
        return None

    return results


def bin_recent_steps_mean(daily_totals: list[int]) -> str:
    """Bin mean daily steps into low/moderate/high.

    Thresholds: <4000=low, 4000-7000=moderate, >7000=high.
    """
    if not daily_totals:
        return "moderate"

    mean_steps = sum(daily_totals) / len(daily_totals)
    if mean_steps < _STEPS_LOW_UPPER:
        return "low"
    if mean_steps > _STEPS_HIGH_LOWER:
        return "high"
    return "moderate"


def bin_walk_pattern(daily_totals: list[int]) -> str:
    """Bin walk pattern into low/high based on mean daily steps.

    Threshold: <5000=low, >=5000=high.
    """
    if not daily_totals:
        return "low"

    mean_steps = sum(daily_totals) / len(daily_totals)
    return "high" if mean_steps >= _WALK_PATTERN_HIGH_THRESHOLD else "low"


def bin_morning_ratio(  # noqa: PLR0911
    morning_steps: list[int], total_steps: list[int]
) -> str:
    """Bin morning_steps_ratio into morning/balanced/evening.

    Ratio = mean(morning) / mean(total).
    Thresholds: <0.4=morning, 0.4-0.6=balanced, >0.6=evening.
    """
    if not morning_steps or not total_steps:
        return "balanced"

    total_morning = sum(morning_steps)
    total_all = sum(total_steps)

    if total_all == 0:
        return "balanced"

    ratio = total_morning / total_all
    if ratio < _MORNING_RATIO_LOW:
        return "morning"
    if ratio > _MORNING_RATIO_HIGH:
        return "evening"
    return "balanced"


def history_to_factors(
    history: list[dict[str, int]],
) -> dict[str, str]:
    """Convert a 7-day history to state factor bins.

    Parameters
    ----------
    history : list[dict]
        List of dicts with keys: day, morning_steps, afternoon_steps.

    Returns
    -------
    dict with keys: recent_steps_mean, recent_walk_pattern, morning_steps_ratio.
    """
    morning_steps = [d["morning_steps"] for d in history]
    afternoon_steps = [d["afternoon_steps"] for d in history]
    daily_totals = [m + a for m, a in zip(morning_steps, afternoon_steps, strict=True)]

    return {
        "recent_steps_mean": bin_recent_steps_mean(daily_totals),
        "recent_walk_pattern": bin_walk_pattern(daily_totals),
        "morning_steps_ratio": bin_morning_ratio(morning_steps, daily_totals),
    }
