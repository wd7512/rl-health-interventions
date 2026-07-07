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
    max_tokens: int = 128,
    max_workers: int = 100,
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
        max_tokens=max_tokens,
        max_workers=max_workers,
    )

    results: list[dict[str, Any]] = []
    for i, resp in enumerate(responses):
        if isinstance(resp, Exception):
            logger.warning("Request %d failed: %s", i + 1, resp)
            results.append({"error": str(resp)})
            continue
        content = _extract_content(resp)
        results.append(
            {"content": content} if content else {"error": "empty"}
        )

    ok = sum(1 for r in results if "content" in r)
    logger.info("Done: %d/%d succeeded", ok, n)
    return results


def save_jsonl(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for i, r in enumerate(results):
            f.write(json.dumps({"index": i, **r}) + "\n")
    logger.info("Wrote %d records to %s", len(results), path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    system_prompt, prompts = generate_prompts()
    logger.info("Generated %d prompts from sprint1", len(prompts))

    if "--dry-run" in sys.argv:
        for i, m in enumerate(build_messages(prompts, system_prompt)):
            print(json.dumps({"index": i, "messages": m}, indent=2))
        return

    results = batch_complete(prompts, system_prompt=system_prompt)
    save_jsonl(results, Path("results.jsonl"))


if __name__ == "__main__":
    main()
