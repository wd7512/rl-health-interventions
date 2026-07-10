"""Shared utilities for LLM bootstrapping request modules."""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

MODEL = "openrouter/deepseek/deepseek-v4-flash"
BASE_URL = "https://openrouter.ai/api/v1"


def resolve_api_key() -> str:
    """Return OPENROUTER_API_KEY or raise RuntimeError."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        msg = "OPENROUTER_API_KEY not set"
        raise RuntimeError(msg)
    return key


def setup_logging() -> None:
    """Configure root logger for CLI entry points."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_env() -> None:
    """Load .env from the llm_bootstrapping directory if present."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


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


def model_short_name() -> str:
    """Derive the short model name from MODEL (e.g. 'deepseek-v4-flash')."""
    return MODEL.rstrip("/").split("/")[-1]


def save_jsonl(
    results: list[dict[str, Any]],
    path: Path,
    *,
    indices: list[int] | None = None,
    offset: int = 0,
) -> None:
    """Append results as JSONL. Use explicit indices or fall back to offset+i."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for i, r in enumerate(results):
            idx = indices[i] if indices is not None else offset + i
            f.write(json.dumps({"index": idx, **r}) + "\n")
    logger = logging.getLogger(__name__)
    logger.info("Wrote %d records to %s", len(results), path)


def dump_messages(prompts: list[str], system_prompt: str | None = None) -> None:
    """Log all messages for dry-run inspection."""
    logger = logging.getLogger(__name__)
    for i, m in enumerate(build_messages(prompts, system_prompt)):
        logger.info(json.dumps({"index": i, "messages": m}, indent=2))


def parse_persona(sys_args: list[str]) -> str:
    """Extract --persona= value from CLI args, default to 'base'."""
    for arg in sys_args:
        if arg.startswith("--persona="):
            return arg.split("=", 1)[1]
    return "base"


def parse_concurrency(sys_args: list[str]) -> int:
    """Extract --concurrency= value from CLI args, default to 200."""
    for arg in sys_args:
        if arg.startswith("--concurrency="):
            return int(arg.split("=", 1)[1])
    return 200


def generate_output_path(persona: str) -> Path:
    """Generate output path from persona, model, and timestamp."""
    timestamp = datetime.datetime.now().strftime("%H:%M_%d:%m:%y")
    return Path(
        f"data/bootstrap/results_{persona}_{model_short_name()}_{timestamp}.jsonl"
    )


def find_latest_results_path(persona: str) -> Path | None:
    """Find the most recent results file for a persona."""
    bootstrap_dir = Path("data/bootstrap")
    if not bootstrap_dir.exists():
        return None
    pattern = f"results_{persona}_{model_short_name()}_*.jsonl"
    files = list(bootstrap_dir.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)
