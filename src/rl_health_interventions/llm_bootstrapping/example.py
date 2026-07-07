"""Standalone litellm example — batch completion via OpenRouter."""

import os
from pathlib import Path

from dotenv import load_dotenv
from litellm import batch_completion

load_dotenv(Path(__file__).parent / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"
)

os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENROUTER_API_BASE"] = OPENROUTER_BASE_URL

MODEL = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"

messages_list = [
    [{"role": "user", "content": "Write a short poem"}] for _ in range(10)
]

responses = batch_completion(
    model=MODEL,
    messages=messages_list,
    base_url=OPENROUTER_BASE_URL,
    max_workers=10,
)

out_path = Path(__file__).parent / "example_results.jsonl"
with out_path.open("w", encoding="utf-8") as f:
    for i, r in enumerate(responses):
        if isinstance(r, Exception):
            f.write(f'{{"index": {i}, "error": "{r}"}}\n')
        else:
            content = r.choices[0].message.content or ""
            f.write(
                f'{{"index": {i}, "content": {content!r}}}\n'
            )

print(f"Wrote {len(responses)} responses to {out_path}")
