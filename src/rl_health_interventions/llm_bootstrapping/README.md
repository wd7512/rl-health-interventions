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
uv run python -m rl_health_interventions.llm_bootstrapping.request --dry-run

# Real run — sends to OpenRouter, writes results.jsonl
uv run python -m rl_health_interventions.llm_bootstrapping.request
```

## Defaults

- Model: `deepseek/deepseek-v4-flash`
- Workers: 50
- Retries: 7
- Temperature: 0.7

## Files

- `request.py` — batch completion via litellm
- `prompts/sprint1.py` — sprint 1 prompt definitions (756 prompts)
- `example.py` — standalone litellm example
- `.example.env` — env template

## Programmatic use

```python
from rl_health_interventions.llm_bootstrapping.request import batch_complete, save_jsonl
from pathlib import Path

results = batch_complete(
    ["What is 2+2?", "Say hello"],
    system_prompt="You are a helpful assistant.",
)
save_jsonl(results, Path("results.jsonl"))
```
