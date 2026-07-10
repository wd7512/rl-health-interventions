"""LLM batch completion via litellm for OpenRouter."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from litellm import batch_completion

from rl_health_interventions.llm_bootstrapping._shared import (
    BASE_URL,
    MODEL,
    build_messages,
    dump_messages,
    generate_output_path,
    load_env,
    model_short_name,
    parse_concurrency,
    parse_persona,
    resolve_api_key,
    save_jsonl,
    setup_logging,
)
from rl_health_interventions.llm_bootstrapping.prompts import generate_prompts

logger = logging.getLogger(__name__)


def check_model_match(out_path: Path) -> None:
    """Raise RuntimeError if the output filename's model doesn't match MODEL."""
    expected = model_short_name()
    stem = out_path.stem
    if not stem.startswith("results_"):
        return
    parts = stem[len("results_") :].split("_")
    min_parts_for_model = 2
    if len(parts) < min_parts_for_model:
        return
    actual = parts[1]
    if actual != expected:
        msg = (
            f"Filename '{out_path.name}' is for model '{actual}' "
            f"but MODEL is '{expected}'."
        )
        raise RuntimeError(msg)


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
    model: str = MODEL,
    temperature: float = 0.7,
    max_workers: int = 50,
    num_retries: int = 7,
) -> list[dict[str, Any]]:
    """Send batch completions to OpenRouter via litellm."""
    api_key = resolve_api_key()
    os.environ["OPENROUTER_API_KEY"] = api_key
    os.environ["OPENROUTER_API_BASE"] = BASE_URL

    messages_list = build_messages(prompts, system_prompt)
    n = len(prompts)
    logger.info("Batch: %d prompts, model=%s, workers=%d", n, model, max_workers)

    responses = batch_completion(
        model=model,
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


def main() -> None:
    max_workers = parse_concurrency(sys.argv)
    chunk_size = max_workers * 10

    setup_logging()
    load_env()

    persona = parse_persona(sys.argv)
    system_prompt, prompts = generate_prompts(persona=persona)
    logger.info("Generated %d prompts for persona=%s", len(prompts), persona)

    if "--dry-run" in sys.argv:
        dump_messages(prompts, system_prompt)
        return

    out_path = generate_output_path(persona)
    total = len(prompts)
    for offset in range(0, total, chunk_size):
        chunk = prompts[offset : offset + chunk_size]
        batch_num = offset // chunk_size + 1
        total_batches = (total + chunk_size - 1) // chunk_size
        logger.info(
            "Batch %d/%d: prompts %d-%d",
            batch_num,
            total_batches,
            offset,
            offset + len(chunk),
        )
        results = batch_complete(
            chunk,
            system_prompt=system_prompt,
            max_workers=max_workers,
        )
        save_jsonl(results, out_path, offset=offset)
    logger.info("All done: %d prompts written to %s", total, out_path)


if __name__ == "__main__":
    main()
