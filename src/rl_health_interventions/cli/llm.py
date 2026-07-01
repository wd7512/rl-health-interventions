"""OpenCode Zen API helper.

Provides a minimal, dependency-free chat completions wrapper around
OpenCode Zen's OpenAI-compatible endpoint.

Usage:
    export OPENCODE_API_KEY=***
    python -m rl_health_interventions.cli.llm --model nemotron-3-ultra-free "Hello"
"""

from __future__ import annotations

import logging
import os
from typing import Any

import urllib.request
import urllib.error
import json

DEFAULT_BASE_URL = "https://opencode.ai/zen/v1"
DEFAULT_MODEL = "nemotron-3-ultra-free"

logger = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    """Raised when the API returns an error or the request fails."""


def get_api_key() -> str:
    """Resolve API key from environment."""
    key = os.environ.get("OPENCODE_API_KEY")
    if not key:
        raise LLMClientError(
            "OPENCODE_API_KEY not set. Export it before calling the API."
        )
    return key


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Send a chat completion request to OpenCode Zen.

    Returns the raw parsed JSON response.
    """
    if api_key is None:
        api_key = get_api_key()

    url = f"{base_url}/chat/completions"
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "User-Agent": "rl-health-interventions/0.1.0",
        },
        method="POST",
    )

    logger.debug("POST %s model=%s", url, model)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise LLMClientError(f"API error {e.code}: {error_body}") from e
    except urllib.error.URLError as e:
        raise LLMClientError(f"Network error: {e.reason}") from e


def chat_text(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Convenience wrapper: send a single user message, return the reply text."""
    resp = chat_completion(
        [{"role": "user", "content": prompt}],
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMClientError(f"Failed to parse API response: {e}") from e


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="OpenCode Zen one-shot chat")
    parser.add_argument("prompt", help="Message to send")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Max completion tokens"
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Sampling temperature"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        reply = chat_text(
            args.prompt,
            model=args.model,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    except LLMClientError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    logger.info("%s", reply)


if __name__ == "__main__":
    main()
