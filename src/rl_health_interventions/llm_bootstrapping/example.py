import os
from pathlib import Path

from dotenv import load_dotenv
from litellm import batch_completion

load_dotenv(Path(__file__).parent / ".env")

# Configure with environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

# Set environment for LiteLLM
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENROUTER_API_BASE"] = OPENROUTER_BASE_URL

MODEL = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"



messages_list = [
    [{"role": "user", "content": "Write a short poem"}] for _ in range(100)
]

responses = batch_completion(
    model=MODEL,
    messages=messages_list,
    base_url=OPENROUTER_BASE_URL,
    max_workers=100,
)

for r in responses:
    if not isinstance(r, Exception):
        print(r.choices[0].message.content)
