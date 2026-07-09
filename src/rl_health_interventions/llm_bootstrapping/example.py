"""Standalone litellm example — batch completion via OpenRouter.

Scratchpad / smoke test — ensures things look right before running the real
bootstrap. Not part of the production pipeline.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from litellm import batch_completion

from rl_health_interventions.llm_bootstrapping.prompts import SYSTEM_PROMPT

load_dotenv(Path(__file__).parent / ".env")

api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

if api_key:
    os.environ["OPENROUTER_API_KEY"] = api_key
if base_url:
    os.environ["OPENROUTER_API_BASE"] = base_url

MODEL = "openrouter/z-ai/glm-5.2"

single_prompt = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": (
            "# Current state\n"
            "It is the evening. Last timestep (afternoon) you were active.\n"
            "Your notification fatigue is high. It is a weekend.\n"
            "Your sleep quality was poor.\n"
            "Your phone prompts you to write in your journal.\n"
            "How many steps do you take this timestep?\n"
            'Output as: {"steps": N, '
            '"step_bin": "inactive"/"moderate"/"active"}'
        ),
    },
]

messages_list = [list(single_prompt) for _ in range(10)]


s = time.time()
responses = batch_completion(
    model=MODEL,
    messages=messages_list,
    base_url=base_url,
    max_workers=5,
    num_retries=3,
)
e = time.time()
print(f"Batch completed in {e - s:.2f} seconds")
print(f"Time per prompt: {(e - s) / len(messages_list):.2f} seconds")
per_prompt = (e - s) / len(messages_list)
print(f"Estimated time for 25000 prompts: {per_prompt * 25000:.2f} seconds")

out_path = Path(__file__).parent / "example_results.jsonl"
with out_path.open("w", encoding="utf-8") as f:
    for i, r in enumerate(responses):
        if isinstance(r, Exception):
            f.write(f'{{"index": {i}, "error": "{r}"}}\n')
        else:
            content = r.choices[0].message.content or ""
            f.write(json.dumps({"index": i, "content": content}) + "\n")

print(f"Wrote {len(responses)} responses to {out_path}")
