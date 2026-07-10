"""Standalone litellm example — batch completion via OpenRouter or OpenCode Zen.

Scratchpad / smoke test — ensures things look right before running the real
bootstrap. Uses the actual prompt rendering from sprint1.py.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

from typing import Any

from litellm import batch_completion

from rl_health_interventions.llm_bootstrapping._shared import (
    BASE_URL,
    MODEL,
    ZEN_BASE_URL,
    ZEN_MODEL,
    load_env,
    parse_provider,
    resolve_api_key,
    resolve_zen_api_key,
)
from rl_health_interventions.llm_bootstrapping.prompts import SYSTEM_PROMPT
from rl_health_interventions.llm_bootstrapping.prompts.sprint1 import (
    _render_day_boundary,
    _render_within_day,
)

load_env()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

provider = parse_provider(sys.argv)

if provider == "zen":
    zen_key = resolve_zen_api_key()
    os.environ["OPENAI_API_KEY"] = zen_key
    model = ZEN_MODEL
    base_url = ZEN_BASE_URL
    extra_kwargs: dict[str, Any] = {
        "api_base": base_url,
        "extra_headers": {"X-API-Key": zen_key},
    }
    logger.info("Provider: zen (model=%s, base=%s)", model, base_url)
else:
    api_key = resolve_api_key()
    os.environ["OPENROUTER_API_KEY"] = api_key
    os.environ["OPENROUTER_API_BASE"] = BASE_URL
    model = MODEL
    extra_kwargs = {}
    logger.info("Provider: openrouter (model=%s)", model)

prompts = [
    _render_day_boundary("active", "low", "weekday", "good"),
    _render_day_boundary("inactive", "high", "weekend", "poor"),
    _render_within_day(0, "active", "low", "movement_suggestion", "weekday", "good"),
    _render_within_day(2, "moderate", "medium", "goal_reminder", "weekend", "poor"),
    _render_within_day(4, "inactive", "high", "journal", "weekday", "good"),
    _render_within_day(1, "active", "low", "idle", "weekday", "good"),
    _render_within_day(3, "inactive", "low", "movement_suggestion", "weekend", "good"),
    _render_within_day(0, "inactive", "medium", "goal_reminder", "weekday", "poor"),
    _render_within_day(2, "active", "high", "movement_suggestion", "weekday", "good"),
    _render_within_day(4, "moderate", "low", "idle", "weekend", "poor"),
]

messages_list = [
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    for prompt in prompts
]

s = time.time()
responses = batch_completion(
    model=model,
    messages=messages_list,
    max_workers=5,
    num_retries=3,
    **extra_kwargs,
)
e = time.time()
logger.info("Batch completed in %.2f seconds", e - s)
logger.info("Time per prompt: %.2f seconds", (e - s) / len(messages_list))
per_prompt = (e - s) / len(messages_list)
logger.info("Estimated time for 22320 prompts: %.2f seconds", per_prompt * 22320)

out_path = Path(__file__).parent / "example_results.jsonl"
with out_path.open("w", encoding="utf-8") as f:
    for i, r in enumerate(responses):
        if isinstance(r, Exception):
            f.write(f'{{"index": {i}, "error": "{r}"}}\n')
        else:
            content = r.choices[0].message.content or ""
            f.write(json.dumps({"index": i, "content": content}) + "\n")

logger.info("Wrote %d responses to %s", len(responses), out_path)
