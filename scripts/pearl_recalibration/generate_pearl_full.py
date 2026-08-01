"""Generate the full 108-state PEARL transition table via LLM bootstrapping.

Full-scale counterpart to generate_pearl_mini.py: 108 states x 13 actions
x samples_per_cell (default 10) = 14,040 calls. Uses the frozen
protocol_fewshot prompt at temperature 0.3 (rounds 14-15 pilot decision).

Processing is chunked over states so the run is resumable and partial
results survive an interruption. Raw records are appended to a single
stable jsonl (state, action, content|error, optional original_content);
the table is aggregated from that raw file at the end.

Usage: uv run python scripts/pearl_recalibration/generate_pearl_full.py
    [--samples N] [--variant NAME] [--temperature T] [--states-per-chunk N]
    [--resume] [--finalize-only]
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl_health_interventions.llm_bootstrapping._shared import (  # noqa: E402
    load_env,
    setup_logging,
)
from rl_health_interventions.llm_bootstrapping.parse_pearl import (  # noqa: E402
    parse_day_history,
)
from rl_health_interventions.llm_bootstrapping.prompts.pearl import (  # noqa: E402
    ACTIONS,
    BURDENS,
    DAY_TYPES,
    MORNING_RATIOS,
    PROMPT_VARIANTS,
    RECENT_STEPS_MEAN,
    WALK_PATTERNS,
    generate_prompts,
)
from rl_health_interventions.llm_bootstrapping.request import (  # noqa: E402
    batch_complete,
)
from rl_health_interventions.llm_bootstrapping.table_aggregate import (  # noqa: E402
    aggregate_to_table,
)

logger = logging.getLogger(__name__)

DEFAULT_SAMPLES = 10
DEFAULT_STATES_PER_CHUNK = 4
TABLE_DIR = _REPO_ROOT / "tables" / "pearl_12action"
RAW_DIR = TABLE_DIR / "raw"


def _state_key(state: dict) -> str:
    return json.dumps(state, sort_keys=True)


def _raw_path(variant: str) -> Path:
    return RAW_DIR / f"results_full_{variant}.jsonl"


def _load_raw_records(raw_path: Path) -> list[dict]:
    """Load raw jsonl records, skipping empty lines and malformed rows."""
    records = []
    with raw_path.open() as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed raw line in %s", raw_path)
    return records


def _completed_states(raw_path: Path) -> set[str]:
    """States whose 13 actions x N samples are already fully recorded."""
    counts: dict[str, dict[str, int]] = {}
    if not raw_path.exists():
        return set()
    for record in _load_raw_records(raw_path):
        key = _state_key(record["state"])
        counts.setdefault(key, {})
        counts[key][record["action"]] = counts[key].get(record["action"], 0) + 1
    completed = set()
    for key, action_counts in counts.items():
        if all(n >= DEFAULT_SAMPLES for n in action_counts.values()) and len(
            action_counts
        ) == len(ACTIONS):
            completed.add(key)
    return completed


def _append_raw(raw_path: Path, records: list[dict]) -> None:
    with raw_path.open("a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _run_chunk(  # noqa: C901, PLR0912
    system_prompt: str,
    chunk: list[dict],
    samples: int,
    temperature: float,
    variant: str,
) -> list[dict]:
    """Run one state chunk: generate prompts, batch-call, bounded retry.

    Returns records in the raw-jsonl shape (state, action, content|error,
    optional original_content).
    """
    prompt_entries = generate_prompts(
        persona="base",
        samples_per_cell=samples,
        state_subset=chunk,
        prompt_variant=variant,
    )[1]

    prompts = [p for p, _s, _a in prompt_entries]
    pairs = [(s, a) for _p, s, a in prompt_entries]
    results = batch_complete(
        prompts,
        system_prompt=system_prompt,
        temperature=temperature,
        max_workers=50,
        provider="openrouter",
    )

    ok = sum(1 for r in results if "content" in r)
    logger.info("Chunk LLM results: %d/%d succeeded", ok, len(results))

    retry_indices = [
        i
        for i, result in enumerate(results)
        if "error" not in result
        and parse_day_history(result.get("content", "")) is None
    ]
    retried_originals: dict[int, dict] = {}
    if retry_indices:
        logger.info("Retrying %d unparseable response(s)", len(retry_indices))
        retry_results = batch_complete(
            [prompts[i] for i in retry_indices],
            system_prompt=system_prompt,
            temperature=temperature,
            max_workers=50,
            provider="openrouter",
        )
        for idx, retry_result in zip(retry_indices, retry_results, strict=True):
            retried_originals[idx] = results[idx]
            results[idx] = retry_result
        n_ok_after = sum(
            1
            for i in retry_indices
            if "content" in results[i]
            and parse_day_history(results[i]["content"]) is not None
        )
        logger.info("Retry recovered %d/%d records", n_ok_after, len(retry_indices))

    records: list[dict] = []
    for idx, (state, action) in enumerate(pairs):
        record: dict = {"state": state, "action": action}
        result = results[idx]
        if "error" in result:
            record["error"] = result["error"]
        else:
            record["content"] = result["content"]
        if idx in retried_originals:
            original = retried_originals[idx]
            if "error" in original:
                record["original_error"] = original["error"]
            else:
                record["original_content"] = original["content"]
        records.append(record)
    return records


def _iter_states() -> list[dict]:
    """All 108 states (same product as prompts.pearl.generate_prompts)."""
    return [
        {
            "recent_steps_mean": rsm,
            "recent_walk_pattern": wp,
            "morning_steps_ratio": mr,
            "day_of_week": dt,
            "burden": b,
        }
        for rsm, wp, mr, dt, b in itertools.product(
            RECENT_STEPS_MEAN, WALK_PATTERNS, MORNING_RATIOS, DAY_TYPES, BURDENS
        )
    ]


def _finalize(raw_path: Path, out_path: Path | None = None) -> Path:
    """Aggregate raw records into the final table."""
    records = _load_raw_records(raw_path)
    if not records:
        msg = f"No raw records in {raw_path}; nothing to aggregate"
        raise ValueError(msg)
    pairs = [(r["state"], r["action"]) for r in records]
    results = [
        {"content": r["content"]} if "content" in r else {"error": r["error"]}
        for r in records
    ]
    table = aggregate_to_table(results, pairs)
    if out_path is None:
        out_path = TABLE_DIR / "pearl_bootstrap.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(table, f, indent=2)
    logger.info(
        "Table has %d transitions; saved to %s", len(table["transitions"]), out_path
    )
    return out_path


def main() -> None:  # noqa: PLR0915
    """Generate the full PEARL transition table."""
    setup_logging()
    load_env()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--variant", default="protocol_fewshot", choices=PROMPT_VARIANTS
    )
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument(
        "--states-per-chunk", type=int, default=DEFAULT_STATES_PER_CHUNK
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip states already fully recorded in the raw file",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Aggregate the raw file into the table without new calls",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = _raw_path(args.variant)

    if args.finalize_only:
        _finalize(raw_path)
        return

    states = _iter_states()
    completed = _completed_states(raw_path) if args.resume else set()
    todo = [s for s in states if _state_key(s) not in completed]
    if args.resume and completed:
        logger.info("Resuming: %d states already recorded", len(completed))
    if not todo:
        logger.info("All states already recorded; finalizing.")
        _finalize(raw_path)
        return

    logger.info(
        "Generating full table (variant=%s, temp=%s): %d states x %d actions "
        "x %d samples = %d calls",
        args.variant,
        args.temperature,
        len(todo),
        len(ACTIONS),
        args.samples,
        len(todo) * len(ACTIONS) * args.samples,
    )

    system_prompt, _ = generate_prompts(
        persona="base",
        samples_per_cell=args.samples,
        state_subset=todo[:1],
        prompt_variant=args.variant,
    )

    start = datetime.now(UTC)
    for i in range(0, len(todo), args.states_per_chunk):
        chunk = todo[i : i + args.states_per_chunk]
        logger.info(
            "Chunk %d/%d: states %d-%d",
            i // args.states_per_chunk + 1,
            (len(todo) + args.states_per_chunk - 1) // args.states_per_chunk,
            i + 1,
            i + len(chunk),
        )
        records = _run_chunk(
            system_prompt,
            chunk,
            args.samples,
            args.temperature,
            args.variant,
        )
        _append_raw(raw_path, records)
        elapsed = (datetime.now(UTC) - start).total_seconds()
        n_done = min(i + len(chunk), len(todo))
        logger.info(
            "Progress: %d/%d states; elapsed %.1f min",
            n_done,
            len(todo),
            elapsed / 60,
        )

    out_path = _finalize(raw_path)
    logger.info("Full-scale run complete: %s", out_path)


if __name__ == "__main__":
    main()
