"""LLM batch completion helper with resume/retry support.

Wraps request.py's batch_complete/save_jsonl with --resume and --retry-errors
flags. Smaller batch sizes (300 vs 2000) for more frequent saves.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from rl_health_interventions.llm_bootstrapping._shared import (
    dump_messages,
    find_latest_results_path,
    generate_output_path,
    load_env,
    parse_concurrency,
    parse_persona,
    parse_subdir,
    save_jsonl,
    setup_logging,
)
from rl_health_interventions.llm_bootstrapping.prompts import generate_prompts
from rl_health_interventions.llm_bootstrapping.request import (
    batch_complete,
    check_model_match,
)

logger = logging.getLogger(__name__)

CHUNK_MULTIPLIER = 1.5


def _load_existing_results(path: Path) -> list[dict[str, Any]]:
    """Load existing JSONL results, skipping malformed lines."""
    if not path.exists():
        return []
    results: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                results.append(json.loads(stripped))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line in %s", path)
    return results


def _resolve_run_mode(
    sys_args: list[str], out_path: Path, total: int
) -> tuple[set[int], list[dict[str, Any]]]:
    """Determine which prompt indices to process based on CLI flags.

    Returns (to_process, existing_records) where existing_records are
    the clean records from a previous run (errors stripped for retry-errors).
    """
    resume = "--resume" in sys_args
    retry_errors = "--retry-errors" in sys_args

    if not resume and not retry_errors:
        if out_path.exists():
            logger.info(
                "Warning: %s exists. Use --resume or --retry-errors. "
                "Starting from scratch.",
                out_path,
            )
        return set(range(total)), []

    existing = _load_existing_results(out_path)
    if not existing:
        return set(range(total)), []

    succeeded = {r["index"] for r in existing if "content" in r and "index" in r}
    error_indices = {r["index"] for r in existing if "error" in r and "index" in r}

    if retry_errors:
        clean = [r for r in existing if r.get("index") not in error_indices]
        logger.info("Retry-errors: re-running %d prompts", len(error_indices))
        return error_indices, clean

    to_process = set(range(total)) - succeeded
    kept = [r for r in existing if "content" in r]
    logger.info(
        "Resume: %d/%d done, running %d",
        len(succeeded),
        total,
        len(to_process),
    )
    return to_process, kept


def main() -> None:
    max_workers = parse_concurrency(sys.argv)
    chunk_size = int(max_workers * CHUNK_MULTIPLIER)
    subdir = parse_subdir(sys.argv)

    setup_logging()
    load_env()

    persona = parse_persona(sys.argv)
    system_prompt, prompts = generate_prompts(persona=persona)
    total = len(prompts)
    logger.info("Generated %d prompts for persona=%s", total, persona)

    if "--dry-run" in sys.argv:
        dump_messages(prompts, system_prompt)
        return

    if "--resume" in sys.argv or "--retry-errors" in sys.argv:
        latest = find_latest_results_path(persona, subdir=subdir)
        if latest is None:
            logger.error("No existing results found for persona=%s", persona)
            return
        out_path = latest
    else:
        out_path = generate_output_path(persona, subdir=subdir)

    if "--resume" in sys.argv or "--retry-errors" in sys.argv:
        check_model_match(out_path)

    to_process, existing_records = _resolve_run_mode(sys.argv, out_path, total)

    if not to_process:
        logger.info("Nothing to process.")
        return

    if existing_records:
        with out_path.open("w", encoding="utf-8") as f:
            for r in existing_records:
                f.write(json.dumps(r) + "\n")

    sorted_indices = sorted(to_process)
    total_to_process = len(sorted_indices)
    logger.info("Processing %d prompts (chunk_size=%d)", total_to_process, chunk_size)

    for batch_start in range(0, total_to_process, chunk_size):
        batch_indices = sorted_indices[batch_start : batch_start + chunk_size]
        batch_num = batch_start // chunk_size + 1
        total_batches = (total_to_process + chunk_size - 1) // chunk_size
        batch_prompts = [prompts[i] for i in batch_indices]
        logger.info(
            "Batch %d/%d: %d prompts (index %d-%d)",
            batch_num,
            total_batches,
            len(batch_prompts),
            batch_indices[0],
            batch_indices[-1],
        )
        results = batch_complete(
            batch_prompts,
            system_prompt=system_prompt,
            max_workers=max_workers,
        )
        save_jsonl(results, out_path, indices=batch_indices)

    all_records = _load_existing_results(out_path)
    all_records.sort(key=lambda r: r.get("index", -1))

    present = {r["index"] for r in all_records if "index" in r}
    if len(present) != total:
        missing = sorted(set(range(total)) - present)
        logger.warning(
            "MISSING %d indices (first 10): %s",
            len(missing),
            missing[:10],
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    ok = sum(1 for r in all_records if "content" in r)
    logger.info(
        "Done: %d/%d results written to %s",
        ok,
        len(all_records),
        out_path,
    )


if __name__ == "__main__":
    main()
