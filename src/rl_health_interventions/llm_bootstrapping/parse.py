"""Parsing stubs for LLM bootstrap responses.

Minimal module — provides bin thresholds and parse stubs. Filled in as
Algorithm 2 (resolved-decisions-sprint-1.md) progresses from raw JSONL to
normalised transition tables.
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Per-timestep thresholds (D1 — resolved-decisions-sprint-1.md)
_INACTIVE_UPPER = 800
_MODERATE_UPPER = 1600


def bin_step_count(steps: int) -> str:
    """Bin a raw step count per D1 per-timestep thresholds.

    <800 → inactive, 800-1600 → moderate, >1600 → active.
    """
    if steps < _INACTIVE_UPPER:
        return "inactive"
    if steps <= _MODERATE_UPPER:
        return "moderate"
    return "active"


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from text.

    Handles LLM responses that wrap JSON in markdown, leading/trailing text, etc.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON: %s", text[:120])
        return None


def parse_day_boundary(response: str) -> str | None:
    """Parse day-boundary LLM response.

    Expected format (shown in prompt):
        {"sleep_quality": "good"} or {"sleep_quality": "poor"}

    Returns ``"good"``, ``"poor"``, or ``None`` on parse failure.
    """
    parsed = extract_json(response)
    if parsed is None:
        return None
    sleep = parsed.get("sleep_quality")
    if sleep not in ("good", "poor"):
        logger.warning("Unexpected sleep_quality value: %s", sleep)
        return None
    return sleep


def parse_within_day(response: str) -> tuple[int, str] | None:
    """Parse within-day LLM response.

    Expected format (shown in prompt):
        {"steps": N, "step_bin": "inactive"/"moderate"/"active"}

    Returns ``(step_count, step_bin)`` or ``None`` on parse failure.
    Validates that step_bin matches the step count per D1 thresholds.
    """
    parsed = extract_json(response)
    if parsed is None:
        return None
    steps = parsed.get("steps")
    step_bin = parsed.get("step_bin")
    if not isinstance(steps, int) or steps < 0:
        logger.warning("Invalid steps value: %s", steps)
        return None
    if step_bin not in ("inactive", "moderate", "active"):
        logger.warning("Unexpected step_bin value: %s", step_bin)
        return None
    computed = bin_step_count(steps)
    if computed != step_bin:
        logger.warning(
            "Step bin mismatch: %d -> computed=%s, claimed=%s",
            steps,
            computed,
            step_bin,
        )
    return steps, computed
