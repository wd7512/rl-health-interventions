#!/usr/bin/env python3
"""Algorithm 2: Bootstrap transition probability tables from an LLM.

Iterates over all (s, a) pairs per table, calls the LLM M times per pair,
aggregates responses into empirical frequencies, and writes 6 JSON files.

Tables produced:
    day_boundary.json       P(sleep' | daily bin, burden, day, sleep)
    within_day_0..4.json    P(step_bin' | step_bin, burden, action, day, sleep)

Usage:
    export OPENCODE_API_KEY=***
    python scripts/llm_bootstrap_transitions.py
    python scripts/llm_bootstrap_transitions.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rl_health_interventions.cli.llm import LLMClientError, chat_completion, get_api_key

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dimension definitions
# ---------------------------------------------------------------------------

STEP_BINS = ["inactive", "moderate", "active"]
BURDENS = ["low", "medium", "high"]
ACTIONS = ["idle", "movement_suggestion", "goal_reminder", "journal"]
DAY_TYPES = ["weekday", "weekend"]
SLEEP_TYPES = ["good", "poor"]

TIMESTEP_NAMES = [
    ("morning", None),
    ("mid-morning", "morning"),
    ("lunch", "mid-morning"),
    ("afternoon", "lunch"),
    ("evening", "afternoon"),
]

_NOTIFICATION_SEQUENCES: dict[str, str] = {
    "low": "idle, idle, idle, idle, idle",
    "medium": "movement_suggestion, idle, idle, idle, idle",
    "high": "movement_suggestion, goal_reminder, idle, idle, idle",
}

_ACTION_SENTENCES: dict[str, str] = {
    "idle": "",
    "movement_suggestion": "Your phone buzzes with a movement suggestion.",
    "goal_reminder": "Your phone reminds you of your step goal.",
    "journal": "Your phone prompts you to write in your journal.",
}

_BIN_MIDPOINTS: dict[str, int] = {
    "inactive": 400,
    "moderate": 1200,
    "active": 2000,
}

# ---------------------------------------------------------------------------
# Prompt templates (from resolved-decisions-sprint-1.md)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Rules: steps <800=inactive, 800-1600=moderate, >1600=active per timestep. "  # noqa: E501
    "Daily <4k=inactive, 4-8k=moderate, >8k=active. "
    "Sleep=good|poor. Burden=low|medium|high."
)

WITHIN_DAY_PROMPT_TIMESTEP_0 = (
    "# Current state\n"
    "You just woke up. It is the morning. It is a {day_type}.\n"
    "Your sleep quality was {sleep}. Your notification fatigue is {burden}.\n"
    "{action_sentence}"
    "How many steps do you take this timestep?"
)

WITHIN_DAY_PROMPT_TIMESTEP_N = (
    "# Current state\n"
    "It is the {time_name}. Last timestep ({prev_time_name}) you were {prev_step_bin}.\n"  # noqa: E501
    "Your notification fatigue is {burden}. It is a {day_type}.\n"
    "Your sleep quality was {sleep}.\n"
    "{action_sentence}"
    "How many steps do you take this timestep?"
)

DAY_BOUNDARY_PROMPT = (
    "# Current state\n"
    "You are at the end of the day. It was a {day_type}.\n"
    "Your sleep quality last night was {sleep}. Your notification fatigue is {burden}.\n\n"  # noqa: E501
    "Today you did ~{steps_estimate} total steps ({step_bin_daily}).\n"
    "Your notifications today: {notification_sequence}\n\n"
    "It's bedtime. How was your sleep quality tonight?\n"
    '{{"sleep_quality": "good"}}\n'
    '{{"sleep_quality": "poor"}}'
)

# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _render_within_day_prompt(
    timestep: int,
    step_bin: str,
    burden: str,
    action: str,
    day_type: str,
    sleep: str,
) -> str:
    time_name, prev_time_name = TIMESTEP_NAMES[timestep]
    action_sentence = _ACTION_SENTENCES[action]
    if timestep == 0:
        return WITHIN_DAY_PROMPT_TIMESTEP_0.format(
            day_type=day_type,
            sleep=sleep,
            burden=burden,
            action_sentence=action_sentence,
        )
    return WITHIN_DAY_PROMPT_TIMESTEP_N.format(
        time_name=time_name,
        prev_time_name=prev_time_name,
        prev_step_bin=step_bin,
        burden=burden,
        day_type=day_type,
        sleep=sleep,
        action_sentence=action_sentence,
    )


def _render_day_boundary_prompt(
    step_bin_daily: str,
    burden: str,
    day_type: str,
    sleep: str,
) -> str:
    midpoint = _BIN_MIDPOINTS.get(step_bin_daily, 4000)
    steps_estimate = midpoint * 5
    seq = _NOTIFICATION_SEQUENCES.get(burden, "idle, idle, idle, idle, idle")
    return DAY_BOUNDARY_PROMPT.format(
        day_type=day_type,
        sleep=sleep,
        burden=burden,
        steps_estimate=steps_estimate,
        step_bin_daily=step_bin_daily,
        notification_sequence=seq,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_step_count(response: str) -> str:
    numbers = re.findall(r"\d+", response)
    if not numbers:
        msg = f"Could not parse step count from: {response!r}"
        raise ValueError(msg)
    count = int(numbers[0])
    if count < 800:
        return "inactive"
    if count <= 1600:
        return "moderate"
    return "active"


def _parse_sleep_quality(response: str) -> str:
    text = response.strip()
    if text.startswith("```"):
        parts = text.split("```", 2)
        text = parts[1] if len(parts) >= 2 else ""
        text = text.removeprefix("json").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group())
            except json.JSONDecodeError:
                msg = f"Could not parse JSON from: {response!r}"
                raise ValueError(msg) from None
        else:
            msg = f"Could not parse JSON from: {response!r}"
            raise ValueError(msg) from None
    quality = obj.get("sleep_quality", "")
    if quality not in ("good", "poor"):
        msg = f"Invalid sleep_quality {quality!r}: {response!r}"
        raise ValueError(msg)
    return str(quality)


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

_SYSTEM_MESSAGE = {"role": "system", "content": SYSTEM_PROMPT}

# Reasoning models (deepseek-v4-flash, etc.) allocate tokens to internal
# thinking before producing output.  Give them enough headroom so the
# actual reply isn't starved.
_MAX_TOKENS = 128
_MAX_RETRIES = 5


def _call_llm(
    prompt: str,
    model: str,
    temperature: float,
    base_url: str,
    api_key: str,
) -> str:
    messages = [_SYSTEM_MESSAGE, {"role": "user", "content": prompt}]
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            body = chat_completion(
                messages=messages,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                max_tokens=_MAX_TOKENS,
            )
        except LLMClientError as e:
            if "429" in str(e):
                wait = 30 if attempt < 3 else 120
                logger.warning(
                    "Rate limited (attempt %d), waiting %ds", attempt + 1, wait
                )
                time.sleep(wait)
                last_error = e
                continue
            raise
        try:
            msg = body["choices"][0]["message"]
            content = str(msg.get("content") or "")
            # Reasoning models (deepseek-v4-flash, etc.) may put the
            # answer in reasoning_content when content is empty.
            if not content.strip():
                reasoning = str(msg.get("reasoning_content") or "")
                # Try to extract the answer from reasoning:
                # 1. JSON pattern {"sleep_quality": "good"}
                # 2. Bare number (only if no JSON in reasoning — avoids
                #    grabbing step-count numbers from day-boundary reasoning)
                for candidate in [reasoning, content]:
                    match = re.search(r"\{[^}]+\}", candidate)
                    if match:
                        content = match.group()
                        break
                if not content.strip() and "{" not in reasoning:
                    match = re.search(r"\b(\d{1,5})\b", reasoning)
                    if match:
                        content = match.group(1)
        except (KeyError, IndexError, TypeError) as e:
            msg_err = f"Failed to parse API response: {e}"
            raise LLMClientError(msg_err) from e
        if content.strip():
            return content
        # Reasoning model used all tokens for thinking — retry
        last_error = LLMClientError(
            f"Empty content on attempt {attempt + 1}/{_MAX_RETRIES}"
        )
        logger.warning(
            "Empty content from %s (attempt %d), retrying", model, attempt + 1
        )
    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def _build_day_boundary_table(
    model: str,
    temperature: float,
    base_url: str,
    api_key: str,
    samples_per_pair: int,
    workers: int,
    dry_run: bool = False,
) -> dict[str, dict[str, float]]:
    day_bins = STEP_BINS if not dry_run else STEP_BINS[:1]
    burd = BURDENS if not dry_run else BURDENS[:1]
    days = DAY_TYPES if not dry_run else DAY_TYPES[:1]
    sleeps = SLEEP_TYPES if not dry_run else SLEEP_TYPES[:1]
    table: dict[str, dict[str, float]] = {}
    tasks: list[dict[str, Any]] = []
    for step_bin_daily in day_bins:
        for burden in burd:
            for day_type in days:
                for sleep in sleeps:
                    key = f"{step_bin_daily},{burden},{day_type},{sleep}"
                    prompt = _render_day_boundary_prompt(
                        step_bin_daily, burden, day_type, sleep
                    )
                    for _ in range(samples_per_pair):
                        tasks.append(
                            {
                                "key": key,
                                "prompt": prompt,
                                "parser": _parse_sleep_quality,
                            }
                        )
    counts: dict[str, dict[str, int]] = {}
    for task in tasks:
        key = task["key"]
        if key not in counts:
            counts[key] = {"good": 0, "poor": 0}

    def _run(task: dict[str, Any]) -> tuple[str, str]:
        response = _call_llm(task["prompt"], model, temperature, base_url, api_key)
        return task["key"], task["parser"](response)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run, t) for t in tasks]
        total = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                key, parsed = fut.result()
                counts[key][parsed] += 1
            except Exception:
                logger.exception("Day-boundary LLM call %d/%d failed", i, total)
                raise
            if i % 50 == 0 or i == total:
                logger.info("Day-boundary: %d/%d LLM calls complete", i, total)
    for key, counter in counts.items():
        total = sum(counter.values())
        table[key] = {outcome: count / total for outcome, count in counter.items()}
    return table


def _build_within_day_table(
    timestep: int,
    model: str,
    temperature: float,
    base_url: str,
    api_key: str,
    samples_per_pair: int,
    workers: int,
    dry_run: bool = False,
) -> dict[str, dict[str, float]]:
    bins = STEP_BINS if not dry_run else STEP_BINS[:1]
    burd = BURDENS if not dry_run else BURDENS[:1]
    acts = ACTIONS if not dry_run else ACTIONS[:2]
    days = DAY_TYPES if not dry_run else DAY_TYPES[:1]
    sleeps = SLEEP_TYPES if not dry_run else SLEEP_TYPES[:1]
    table: dict[str, dict[str, float]] = {}
    tasks: list[dict[str, Any]] = []
    for step_bin in bins:
        for burden in burd:
            for action in acts:
                for day_type in days:
                    for sleep in sleeps:
                        key = f"{step_bin},{burden},{action},{day_type},{sleep}"
                        prompt = _render_within_day_prompt(
                            timestep, step_bin, burden, action, day_type, sleep
                        )
                        for _ in range(samples_per_pair):
                            tasks.append(
                                {
                                    "key": key,
                                    "prompt": prompt,
                                    "parser": _parse_step_count,
                                }
                            )
    counts: dict[str, dict[str, int]] = {}
    for task in tasks:
        key = task["key"]
        if key not in counts:
            counts[key] = {"inactive": 0, "moderate": 0, "active": 0}

    def _run(task: dict[str, Any]) -> tuple[str, str]:
        response = _call_llm(task["prompt"], model, temperature, base_url, api_key)
        return task["key"], task["parser"](response)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run, t) for t in tasks]
        total = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                key, parsed = fut.result()
                counts[key][parsed] += 1
            except Exception:
                logger.exception(
                    "Within-day %d LLM call %d/%d failed", timestep, i, total
                )
                raise
            if i % 100 == 0 or i == total:
                logger.info("Within-day %d: %d/%d calls complete", timestep, i, total)
    for key, counter in counts.items():
        total = sum(counter.values())
        table[key] = {outcome: count / total for outcome, count in counter.items()}
    return table


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def _write_table(table: dict[str, dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info("Wrote %s (%d entries)", path, len(table))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap transition tables from an LLM (Alg 2)"
    )
    parser.add_argument(
        "--model", default="deepseek-v4-flash-free", help="LLM model ID"
    )
    parser.add_argument(
        "--base-url", default="https://opencode.ai/zen/v1", help="API base URL"
    )
    parser.add_argument(
        "--output-dir", default="tables", help="Output directory for JSON files"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument("--workers", type=int, default=100, help="Concurrent workers")
    parser.add_argument(
        "--dry-run", action="store_true", help="Minimal pairs to verify format"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable DEBUG logging"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        api_key = get_api_key()
    except LLMClientError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    if args.dry_run:
        samples_per_pair = 2
        logger.info("DRY-RUN: %d samples/pair, limited (s,a) pairs", samples_per_pair)
    else:
        samples_per_pair = 30
    output_dir = Path(args.output_dir)
    logger.info("Building day-boundary table...")
    db_samples = 1 if args.dry_run else 20
    day_boundary = _build_day_boundary_table(
        model=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
        api_key=api_key,
        samples_per_pair=db_samples,
        workers=args.workers,
        dry_run=args.dry_run,
    )
    _write_table(day_boundary, output_dir / "day_boundary.json")
    for timestep in range(5):
        logger.info("Building within-day table %d...", timestep)
        table = _build_within_day_table(
            timestep=timestep,
            model=args.model,
            temperature=args.temperature,
            base_url=args.base_url,
            api_key=api_key,
            samples_per_pair=samples_per_pair,
            workers=args.workers,
            dry_run=args.dry_run,
        )
        _write_table(table, output_dir / f"within_day_{timestep}.json")
    logger.info("Bootstrap complete. 6 tables written to %s/", output_dir)


if __name__ == "__main__":
    main()
