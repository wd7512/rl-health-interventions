# LLM Bootstrapping

Batch LLM completions via [litellm](https://github.com/BerriAI/litellm) for OpenRouter.

## Setup

```bash
export OPENROUTER_API_KEY=sk-or-v1-***
```

Or create a `.env` file in this directory:

```
OPENROUTER_API_KEY=sk-or-v1-***
OPENCODE_ZEN_API_KEY=your_opencode_zen_api_key_here
```

`OPENCODE_ZEN_API_KEY` is only required when using `--provider=zen`.

## Usage

```bash
# Dry run — preview messages without API calls
uv run python -m rl_health_interventions.llm_bootstrapping.request --dry-run

# Real run — sends to OpenRouter, writes results_<model>_<timestamp>.jsonl
uv run python -m rl_health_interventions.llm_bootstrapping.request

# Resume — skip already-succeeded prompts, append new results, sort on finish
uv run python -m rl_health_interventions.llm_bootstrapping.request_helper --resume

# Retry errors — strip error records, re-run them, append, sort on finish
uv run python -m rl_health_interventions.llm_bootstrapping.request_helper --retry-errors

# Use OpenCode Zen provider (requires OPENCODE_ZEN_API_KEY in .env)
uv run python -m rl_health_interventions.llm_bootstrapping.request --provider=zen
uv run python -m rl_health_interventions.llm_bootstrapping.request_helper --retry-errors --persona=stable_maintainer --provider=zen

# Custom output path
uv run python -m rl_health_interventions.llm_bootstrapping.request_helper --resume --output=path/to/file.jsonl
```

`request_helper.py` uses a smaller batch size (300 vs 2000) so results hit disk more frequently.

`--resume` and `--retry-errors` validate the filename's embedded model name against `MODEL` and raise an error on mismatch.

## Defaults

- Model: `deepseek/deepseek-v4-flash`
- Workers: 200
- Retries: 7
- Temperature: 0.7

## Files

- `request.py` — batch completion via litellm (base script); exports `model_short_name()`, `check_model_match()`, `batch_complete()`, `save_jsonl()`
- `request_helper.py` — resume/retry-errors wrapper with smaller batches
- `prompts/sprint1.py` — sprint 1 prompt definitions (22,320 prompts with samples_per_cell=10)
- `example.py` — standalone litellm example
- `.example.env` — env template

## Programmatic use

```python
from rl_health_interventions.llm_bootstrapping.request import (
    batch_complete,
    model_short_name,
    save_jsonl,
)
from pathlib import Path

results = batch_complete(
    ["What is 2+2?", "Say hello"],
    system_prompt="You are a helpful assistant.",
)
save_jsonl(results, Path(f"results_{model_short_name()}_<timestamp>.jsonl"))
```

## Persona support

Run bootstrap with a specific persona system prompt:

```bash
uv run python -m rl_health_interventions.llm_bootstrapping.request --persona=goal_driven
uv run python -m rl_health_interventions.llm_bootstrapping.request_helper --persona=social_responder --concurrency=1000
```

Available personas: `base`, `goal_driven`, `social_responder`, `resistant`, `stable_maintainer`.

Output files: `data/bootstrap/results_{persona}_{model}_{timestamp}.jsonl`

Resume/retry with persona:
```bash
uv run python -m rl_health_interventions.llm_bootstrapping.request_helper --resume --persona=goal_driven
uv run python -m rl_health_interventions.llm_bootstrapping.request_helper --retry-errors --persona=goal_driven
```

## TODO: stable_maintainer incomplete

**CRITICAL:** The `stable_maintainer` persona run has 721 rate-limited errors out of 22,320
prompts (96.6% complete). All 721 failed with `Key limit exceeded (total limit)` from
DeepSeek V4 Flash via OpenRouter. This is a **model-specific rate limit** — other models
like `glm-5.2` work fine (verified via `example.py`).

**To complete (option 1 — OpenCode Zen):**
1. Add `OPENCODE_ZEN_API_KEY` to `.env` (see Setup above)
2. Retry with Zen provider:
   ```bash
   uv run python -m rl_health_interventions.llm_bootstrapping.request_helper --retry-errors --persona=stable_maintainer --provider=zen
   ```

**To complete (option 2 — wait for OpenRouter reset):**
1. Check OpenRouter rate limits for `deepseek-v4-flash` at https://openrouter.ai/workspaces
2. Once limit resets, run:
   ```bash
   uv run python -m rl_health_interventions.llm_bootstrapping.request_helper --retry-errors --persona=stable_maintainer
   ```
3. Verify 0 errors:
   ```bash
   jq -s '[.[] | select(.error)] | length' data/bootstrap/results_stable_maintainer*.jsonl
   ```
