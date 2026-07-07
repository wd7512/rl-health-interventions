import os
from litellm import completion

# Configure with environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

# Set environment for LiteLLM
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENROUTER_API_BASE"] = OPENROUTER_BASE_URL

MODEL = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"  # Specify the model you want to use

test_message = [{"role": "user", "content": "Write a short poem"}]

response = completion(
    model=MODEL,
    messages=test_message,
    base_url=OPENROUTER_BASE_URL  # Explicitly pass base_url for clarity
)