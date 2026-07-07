# LLM Bootstrapping

Batch LLM completions via [litellm](https://github.com/BerriAI/litellm) for OpenRouter.

## Setup

```bash
export OPENROUTER_API_KEY=sk-or-v1-***
```

Or create a `.env` file in this directory:

```
OPENROUTER_API_KEY=sk-or-v1-***
```

## Usage

```bash
# Dry run — preview messages without API calls
echo -e "What is 2+2?\nSay hello" | uv run python -m rl_health_interventions.llm_bootstrapping.request --dry-run

# Real run — sends to OpenRouter, writes results.jsonl
echo -e "What is 2+2?\nSay hello" | uv run python -m rl_health_interventions.llm_bootstrapping.request
```

## Files

- `request.py` — batch completion via litellm
- `example.py` — standalone litellm example
- `.example.env` — env template

## Programmatic use

```python
from rl_health_interventions.llm_bootstrapping.request import batch_complete, save_jsonl
from pathlib import Path

results = batch_complete(
    ["What is 2+2?", "Say hello"],
    system_prompt="You are a helpful assistant.",
    max_workers=100,
)
save_jsonl(results, Path("results.jsonl"))
```
