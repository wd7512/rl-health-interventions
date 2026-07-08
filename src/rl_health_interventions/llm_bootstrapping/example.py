"""Standalone litellm example — batch completion via OpenRouter."""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from litellm import batch_completion

from rl_health_interventions.llm_bootstrapping.prompts.sprint1 import SYSTEM_PROMPT

load_dotenv(Path(__file__).parent / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENROUTER_API_BASE"] = OPENROUTER_BASE_URL

MODEL = "openrouter/z-ai/glm-5.2"

single_prompt = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": '# Current state\nIt is the evening. Last timestep (afternoon) you were active.\nYour notification fatigue is high. It is a weekend.\nYour sleep quality was poor.\nYour phone prompts you to write in your journal.\nHow many steps do you take this timestep?\nOutput as: {"steps": N, "step_bin": "inactive"/"moderate"/"active"}',
    },
]

messages_list = [list(single_prompt) for _ in range(10)]


s = time.time()
responses = batch_completion(
    model=MODEL,
    messages=messages_list,
    base_url=OPENROUTER_BASE_URL,
    max_workers=5,
    num_retries=3,
)
e = time.time()
print(f"Batch completed in {e - s:.2f} seconds")
print(f"Time per prompt: {(e - s) / len(messages_list):.2f} seconds")
print(
    f"Estimated time for 25000 prompts: {(e - s) / len(messages_list) * 25000:.2f} seconds"
)

out_path = Path(__file__).parent / "example_results.jsonl"
with out_path.open("w", encoding="utf-8") as f:
    for i, r in enumerate(responses):
        if isinstance(r, Exception):
            f.write(f'{{"index": {i}, "error": "{r}"}}\n')
        else:
            content = r.choices[0].message.content or ""
            f.write(json.dumps({"index": i, "content": content}) + "\n")

print(f"Wrote {len(responses)} responses to {out_path}")
