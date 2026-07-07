"""Standalone litellm example — batch completion via OpenRouter."""

import os
from pathlib import Path
import time

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

messages_list = [[
    {
      "role": "system",
      "content": "# Reference\n\nYou are a generally healthy adult looking to improve your exercise and sleep habits.\n\n5 timesteps per day: morning, mid-morning, lunch, afternoon, evening\n\nPer-timestep step ranges (daily threshold / 5):\n  <800 steps     = inactive\n  800-1600 steps = moderate\n  >1600 steps    = active\n\nSleep quality: good / poor (based on how well you slept)\n\nDaily step total ranges (5 timesteps x per-timestep ranges):\n  <4000 steps total     = inactive\n  4000-8000 steps total = moderate\n  >8000 steps total     = active\n\nBurden (notification fatigue):\n  low     = 0 of last 3 timesteps had an intervention\n  medium  = 1 of last 3\n  high    = 2 or 3of last 3"
    },
    {
      "role": "user",
      "content": "# Current state\nIt is the evening. Last timestep (afternoon) you were active.\nYour notification fatigue is high. It is a weekend.\nYour sleep quality was poor.\nYour phone prompts you to write in your journal.\nHow many steps do you take this timestep?"
    }] * 10
  ]


s = time.time()
responses = batch_completion(
    model=MODEL,
    messages=messages_list,
    base_url=OPENROUTER_BASE_URL,
    max_workers=5,
)
e = time.time()
print(f"Batch completed in {e - s:.2f} seconds")
print(f"Time per prompt: {(e - s) / len(messages_list):.2f} seconds")
print(f"Estimated time for 25000 prompts: {(e - s) / len(messages_list) * 25000:.2f} seconds")

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
