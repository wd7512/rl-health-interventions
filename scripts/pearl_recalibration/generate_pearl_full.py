"""Generate the full 108-state PEARL transition table via LLM bootstrapping.

Full-scale counterpart to generate_pearl_mini.py: 108 states x 13 actions
x samples_per_cell (default 10) = 14,040 calls. Uses the frozen
protocol_fewshot prompt at temperature 0.3 (rounds 14-15 pilot decision).

The workflow mirrors Sprint 1's request.py + request_helper.py pattern:
raw responses are appended to a single stable jsonl *after every batch*
(rather than after a whole state chunk), and --resume / --retry-errors fill
in the gaps cell-granularly. A stalled request therefore never risks losing
completed work -- everything already on disk survives, and a re-run tops up
only the missing (state, action, sample) cells.

Cell = one (state, action) pair needing `samples` parseable responses. A
cell is complete when the raw file holds >= `samples` records with `content`
(error records do not count).

Usage: uv run python scripts/pearl_recalibration/generate_pearl_full.py
    [--samples N] [--variant NAME] [--temperature T] [--batch-size N]
    [--workers N] [--max-states N] [--resume] [--retry-errors]
    [--finalize-only]
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
    _render_system_prompt,
    _render_user_prompt,
)
from rl_health_interventions.llm_bootstrapping.request import (  # noqa: E402
    batch_complete,
)
from rl_health_interventions.llm_bootstrapping.table_aggregate import (  # noqa: E402
    aggregate_to_table,
)

logger = logging.getLogger(__name__)

DEFAULT_SAMPLES = 10
DEFAULT_BATCH_SIZE = 100
DEFAULT_WORKERS = 50
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


def _append_raw(raw_path: Path, records: list[dict]) -> None:
    with raw_path.open("a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _strip_errors(raw_path: Path) -> None:
    """Rewrite the raw file without error records (used by --retry-errors)."""
    if not raw_path.exists():
        return
    keep = [r for r in _load_raw_records(raw_path) if "content" in r]
    with raw_path.open("w") as f:
        for record in keep:
            f.write(json.dumps(record) + "\n")
    logger.info("Stripped error records; %d content records remain", len(keep))


def _content_counts(raw_path: Path) -> dict[tuple[str, str], int]:
    """Count content records per (state_key, action) cell."""
    counts: dict[tuple[str, str], int] = {}
    for record in _load_raw_records(raw_path):
        if "content" in record:
            key = (_state_key(record["state"]), record["action"])
            counts[key] = counts.get(key, 0) + 1
    return counts


def _todo_cells(raw_path: Path, samples: int) -> list[tuple[dict, str, int]]:
    """Cells needing prompts: [(state, action, needed)].

    Needed = samples - content records already on disk. Error records do not
    count, so a partially-filled cell is topped up to `samples` rather than
    regenerated from scratch.
    """
    counts = _content_counts(raw_path) if raw_path.exists() else {}
    todo: list[tuple[dict, str, int]] = []
    for state in _iter_states():
        key = _state_key(state)
        for action in ACTIONS:
            needed = samples - counts.get((key, action), 0)
            if needed > 0:
                todo.append((state, action, needed))
    return todo


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


def _records_from_results(
    pairs: list[tuple[dict, str]],
    results: list[dict],
    retried_originals: dict[int, dict] | None = None,
) -> list[dict]:
    """Build raw-jsonl records from LLM results.

    retried_originals maps result index -> the original result it replaced,
    so retries keep the original content/error alongside the new one.
    """
    retried_originals = retried_originals or {}
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


def _unparseable_indices(results: list[dict]) -> list[int]:
    """Indices of results that returned content we could not parse."""
    return [
        i
        for i, result in enumerate(results)
        if "error" not in result
        and parse_day_history(result.get("content", "")) is None
    ]


def _run_batch(
    system_prompt: str,
    batch: list[tuple[str, dict, str]],
    temperature: float,
    max_workers: int,
) -> tuple[list[dict], list[tuple[str, dict, str]]]:
    """Call the LLM for one batch; return (records, retry_batch).

    batch is a list of (prompt, state, action). The primary batch runs first
    and its records are returned alongside a retry_batch of unparseable
    responses, so the caller can append the primary records before any retry
    call -- a stalled retry never holds the primary batch hostage in memory.
    """
    prompts = [p for p, _s, _a in batch]
    pairs = [(s, a) for _p, s, a in batch]
    results = batch_complete(
        prompts,
        system_prompt=system_prompt,
        temperature=temperature,
        max_workers=max_workers,
        provider="openrouter",
    )

    ok = sum(1 for r in results if "content" in r)
    logger.info("Batch LLM results: %d/%d succeeded", ok, len(results))

    records = _records_from_results(pairs, results)
    retry_indices = _unparseable_indices(results)
    retry_batch = [(prompts[i], pairs[i][0], pairs[i][1]) for i in retry_indices]
    if retry_batch:
        logger.info("Retrying %d unparseable response(s)", len(retry_batch))
    return records, retry_batch


def _run_retry(
    system_prompt: str,
    retry_batch: list[tuple[str, dict, str]],
    original_records: dict[tuple[str, str], dict],
    temperature: float,
    max_workers: int,
) -> list[dict]:
    """Re-run unparseable prompts; return records with original preserved.

    original_records maps (state_key, action) -> the primary record replaced,
    keyed by the first occurrence so retry output keeps the original.
    """
    prompts = [p for p, _s, _a in retry_batch]
    pairs = [(s, a) for _p, s, a in retry_batch]
    results = batch_complete(
        prompts,
        system_prompt=system_prompt,
        temperature=temperature,
        max_workers=max_workers,
        provider="openrouter",
    )
    retried_originals = {
        i: original_records.get(_state_key(pairs[i][0]) + "||" + pairs[i][1], {})
        for i in range(len(pairs))
    }
    records = _records_from_results(pairs, results, retried_originals)
    n_ok_after = sum(
        1
        for r in records
        if "content" in r and parse_day_history(r["content"]) is not None
    )
    logger.info("Retry recovered %d/%d records", n_ok_after, len(pairs))
    return records


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


def main() -> None:  # noqa: C901, PLR0912, PLR0915
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
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Prompts per LLM batch; raw records are appended after each batch",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--max-states",
        type=int,
        default=0,
        help="Stop after this many states (0 = all remaining states)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Top up cells that already have < --samples responses",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Strip error records, then top up the affected cells",
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

    if raw_path.exists() and not args.resume and not args.retry_errors:
        logger.error(
            "%s already exists; use --resume or --retry-errors "
            "to avoid duplicating cells",
            raw_path,
        )
        sys.exit(1)

    if args.retry_errors:
        _strip_errors(raw_path)

    todo = _todo_cells(raw_path, args.samples)
    if args.max_states:
        seen: set[str] = set()
        filtered: list[tuple[dict, str, int]] = []
        for cell in todo:
            key = _state_key(cell[0])
            if key not in seen:
                if len(seen) >= args.max_states:
                    break
                seen.add(key)
            filtered.append(cell)
        todo = filtered

    if not todo:
        logger.info("All cells already complete; finalizing.")
        _finalize(raw_path)
        return

    total_prompts = sum(needed for _s, _a, needed in todo)
    logger.info(
        "Generating full table (variant=%s, temp=%s): %d cells, %d prompts "
        "(batch_size=%d, workers=%d)",
        args.variant,
        args.temperature,
        len(todo),
        total_prompts,
        args.batch_size,
        args.workers,
    )

    system_prompt = _render_system_prompt("base", args.variant)

    start = datetime.now(UTC)
    done_prompts = 0
    batch: list[tuple[str, dict, str]] = []
    for state, action, needed in todo:
        prompt = _render_user_prompt(
            recent_steps_mean=state["recent_steps_mean"],
            walk_pattern=state["recent_walk_pattern"],
            morning_ratio=state["morning_steps_ratio"],
            day_type=state["day_of_week"],
            burden=state["burden"],
            action=action,
            prompt_variant=args.variant,
        )
        for _ in range(needed):
            batch.append((prompt, state, action))
        if len(batch) >= args.batch_size:
            _flush_batch(
                raw_path,
                system_prompt,
                batch,
                args.temperature,
                args.workers,
                total_prompts,
                done_prompts,
                start,
            )
            done_prompts += len(batch)
            batch = []
    if batch:
        _flush_batch(
            raw_path,
            system_prompt,
            batch,
            args.temperature,
            args.workers,
            total_prompts,
            done_prompts,
            start,
        )

    out_path = _finalize(raw_path)
    logger.info("Full-scale run complete: %s", out_path)


def _flush_batch(
    raw_path: Path,
    system_prompt: str,
    batch: list[tuple[str, dict, str]],
    temperature: float,
    max_workers: int,
    total_prompts: int,
    done_prompts: int,
    start: datetime,
) -> None:
    """Run one batch, appending the primary records before retries.

    The primary batch's records hit disk immediately so a stalled retry call
    never blocks already-succeeded output; the retry batch (if any) appends
    separately once it resolves.
    """
    records, retry_batch = _run_batch(system_prompt, batch, temperature, max_workers)
    _append_raw(raw_path, records)
    done_prompts += len(batch)
    if retry_batch:
        first_originals = {
            (_state_key(s) + "||" + a): rec
            for rec, (_p, s, a) in zip(records, batch, strict=True)
        }
        retry_records = _run_retry(
            system_prompt,
            retry_batch,
            first_originals,
            temperature,
            max_workers,
        )
        _append_raw(raw_path, retry_records)
    elapsed = (datetime.now(UTC) - start).total_seconds()
    logger.info(
        "Progress: %d/%d prompts; elapsed %.1f min",
        done_prompts,
        total_prompts,
        elapsed / 60,
    )

    out_path = _finalize(raw_path)
    logger.info("Full-scale run complete: %s", out_path)


if __name__ == "__main__":
    main()
