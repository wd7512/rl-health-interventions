"""LLM batch completion via litellm for OpenRouter."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from litellm import batch_completion

from rl_health_interventions.llm_bootstrapping.prompts.sprint1 import (
    generate_prompts,
)

logger = logging.getLogger(__name__)

MODEL = "openrouter/deepseek/deepseek-v4-flash"
BASE_URL = "https://openrouter.ai/api/v1"


def _resolve_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        msg = "OPENROUTER_API_KEY not set"
        raise RuntimeError(msg)
    return key


def build_messages(
    prompts: list[str], system_prompt: str | None = None
) -> list[list[dict[str, str]]]:
    """Build message lists from prompts."""
    messages_list: list[list[dict[str, str]]] = []
    for prompt in prompts:
        msgs: list[dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        messages_list.append(msgs)
    return messages_list


def _extract_content(resp: Any) -> str:
    """Extract content from a litellm response, handling reasoning models."""
    try:
        msg = resp.choices[0].message
    except (AttributeError, IndexError):
        return ""
    content = getattr(msg, "content", None) or ""
    if content.strip():
        return content
    reasoning = getattr(msg, "reasoning_content", None) or ""
    return reasoning.strip() if reasoning.strip() else ""


def batch_complete(
    prompts: list[str],
    *,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_workers: int = 50,
    num_retries: int = 7,
) -> list[dict[str, Any]]:
    """Send batch completions to OpenRouter via litellm."""
    api_key = _resolve_api_key()
    os.environ["OPENROUTER_API_KEY"] = api_key
    os.environ["OPENROUTER_API_BASE"] = BASE_URL

    messages_list = build_messages(prompts, system_prompt)
    n = len(prompts)
    logger.info("Batch: %d prompts, workers=%d", n, max_workers)

    responses = batch_completion(
        model=MODEL,
        messages=messages_list,
        temperature=temperature,
        max_workers=max_workers,
        num_retries=num_retries,
    )

    results: list[dict[str, Any]] = []
    for i, resp in enumerate(responses):
        if isinstance(resp, Exception):
            logger.warning("Request %d failed: %s", i + 1, resp)
            results.append({"prompt": prompts[i], "error": str(resp)})
            continue
        content = _extract_content(resp)
        results.append(
            {"prompt": prompts[i], "content": content}
            if content
            else {"prompt": prompts[i], "error": "empty"}
        )

    ok = sum(1 for r in results if "content" in r)
    logger.info("Done: %d/%d succeeded", ok, n)
    return results


def save_jsonl(results: list[dict[str, Any]], path: Path, offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for i, r in enumerate(results):
            f.write(json.dumps({"index": offset + i, **r}) + "\n")
    logger.info("Wrote %d records to %s (offset %d)", len(results), path, offset)


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
    No file I/O — caller handles merge and write.
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

    succeeded = {r["index"] for r in existing if "content" in r}
    error_indices = {r["index"] for r in existing if "error" in r}

    if retry_errors:
        clean = [r for r in existing if r["index"] not in error_indices]
        logger.info("Retry-errors: re-running %d prompts", len(error_indices))
        return error_indices, clean

    # --resume
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
    max_workers = 200
    chunk_size = max_workers * 10

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    system_prompt, prompts = generate_prompts()
    total = len(prompts)
    logger.info("Generated %d prompts from sprint1", total)

    if "--dry-run" in sys.argv:
        for i, m in enumerate(build_messages(prompts, system_prompt)):
            print(json.dumps({"index": i, "messages": m}, indent=2))
        return

    out_path = Path("results.jsonl")
    to_process, existing_records = _resolve_run_mode(sys.argv, out_path, total)

    if not to_process:
        logger.info("Nothing to process.")
        return

    sorted_indices = sorted(to_process)
    total_to_process = len(sorted_indices)
    logger.info("Processing %d prompts", total_to_process)

    new_results: list[dict[str, Any]] = []
    for batch_start in range(0, total_to_process, chunk_size):
        batch_indices = sorted_indices[batch_start : batch_start + chunk_size]
        batch_num = batch_start // chunk_size + 1
        total_batches = (total_to_process + chunk_size - 1) // chunk_size
        batch_prompts = [prompts[i] for i in batch_indices]
        logger.info(
            "Batch %d/%d: %d prompts (index range %d-%d)",
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
        for i, r in enumerate(results):
            new_results.append({"index": batch_indices[i], **r})

    all_results = existing_records + new_results
    all_results.sort(key=lambda r: r["index"])

    present = {r["index"] for r in all_results}
    if len(present) != total:
        missing = sorted(set(range(total)) - present)
        logger.warning(
            "MISSING %d indices (first 10): %s",
            len(missing),
            missing[:10],
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    ok = sum(1 for r in all_results if "content" in r)
    logger.info(
        "Done: %d/%d results written to %s",
        ok,
        len(all_results),
        out_path,
    )


if __name__ == "__main__":
    main()
