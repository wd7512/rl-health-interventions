"""Parsing for PEARL LLM bootstrap responses.

Parses 7-day step history responses and bins raw step counts into
state factors for the 12-action PEARL config.
"""

from __future__ import annotations

import json
import logging

from rl_health_interventions.llm_bootstrapping.prompts.pearl import (
    MORNING_RATIO_HIGH,
    MORNING_RATIO_LOW,
    STEPS_HIGH_LOWER,
    STEPS_LOW_UPPER,
    WALK_PATTERN_HIGH_THRESHOLD,
)

logger = logging.getLogger(__name__)


def parse_day_history(  # noqa: C901, PLR0912, PLR0915
    response: str, expected_days: int = 7
) -> list[dict[str, int]] | None:
    """Parse a 7-day step history from LLM response.

    Expected format: 7 JSON objects, each with day, morning_steps, afternoon_steps.
    The LLM may output these on separate lines or in a single block.

    Returns list of 7 dicts with keys: day, morning_steps, afternoon_steps.
    Returns None unless exactly ``expected_days`` distinct days parse.
    """
    results = []
    dropped = 0
    lines = response.strip().split("\n")

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue

        if not isinstance(obj, dict):
            dropped += 1
            continue

        day = obj.get("day")
        morning = obj.get("morning_steps")
        afternoon = obj.get("afternoon_steps")

        if day is None or morning is None or afternoon is None:
            dropped += 1
            continue

        if (
            not isinstance(day, (int, float))
            or not isinstance(morning, (int, float))
            or not isinstance(afternoon, (int, float))
        ):
            dropped += 1
            continue

        if morning < 0 or afternoon < 0:
            dropped += 1
            continue

        results.append(
            {
                "day": int(day),
                "morning_steps": int(morning),
                "afternoon_steps": int(afternoon),
            }
        )

    if dropped:
        logger.warning("Dropped %d malformed line(s) in response", dropped)

    if len(results) == 0:
        logger.warning("No valid day records found in response")
        return None

    if len(results) != expected_days or len({d["day"] for d in results}) != len(
        results
    ):
        logger.warning(
            "Expected %d distinct day records, got %d (days=%s)",
            expected_days,
            len(results),
            sorted(d["day"] for d in results),
        )
        return None

    return results


def bin_recent_steps_mean(daily_totals: list[int]) -> str:
    """Bin mean daily steps into low/moderate/high.

    Thresholds: <4000=low, 4000-7000=moderate, >7000=high.
    """
    if not daily_totals:
        return "moderate"

    mean_steps = sum(daily_totals) / len(daily_totals)
    if mean_steps < STEPS_LOW_UPPER:
        return "low"
    if mean_steps > STEPS_HIGH_LOWER:
        return "high"
    return "moderate"


def bin_walk_pattern(daily_totals: list[int]) -> str:
    """Bin walk pattern into low/high based on mean daily steps.

    Threshold: <5000=low, >=5000=high.
    """
    if not daily_totals:
        return "low"

    mean_steps = sum(daily_totals) / len(daily_totals)
    return "high" if mean_steps >= WALK_PATTERN_HIGH_THRESHOLD else "low"


def bin_morning_ratio(  # noqa: PLR0911
    morning_steps: list[int], total_steps: list[int]
) -> str:
    """Bin morning_steps_ratio into morning/balanced/evening.

    Ratio = mean(morning) / mean(total).
    A low ratio means most steps occur in the afternoon/evening.
    Thresholds: <0.4=evening, 0.4-0.6=balanced, >0.6=morning.
    """
    if not morning_steps or not total_steps:
        return "balanced"

    total_morning = sum(morning_steps)
    total_all = sum(total_steps)

    if total_all == 0:
        return "balanced"

    ratio = total_morning / total_all
    if ratio < MORNING_RATIO_LOW:
        return "evening"
    if ratio > MORNING_RATIO_HIGH:
        return "morning"
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
