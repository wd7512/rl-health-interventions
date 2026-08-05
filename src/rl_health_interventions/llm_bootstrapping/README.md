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

## Analysis and visualization

```bash
# Analyze LLM bootstrap results
uv run python -m rl_health_interventions.llm_bootstrapping.analyze \
    --files data/bootstrap/results_deepseek.jsonl \
           data/bootstrap/results_glm5.2.jsonl

# Generate transition matrix figures
uv run python -m rl_health_interventions.llm_bootstrapping.visualize \
    --files data/bootstrap/results_deepseek.jsonl \
           data/bootstrap/results_glm5.2.jsonl
```

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
- `analyze.py` — analysis of bootstrap results (parse rates, distributions, transition realism)
- `visualize.py` — transition matrix charts with directed edges
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

## Full-scale PEARL bootstrap (12-action transition table)

The full 108-state PEARL table is generated the same way as Sprint 1: a base
generator writes raw responses to a JSONL, and a resume/retry pass fills the
gaps. The table is **cell-granular**: a cell is one `(state, action)` pair that
needs `--samples` (default 10) parseable responses.

### Workflow

```bash
# 1. Generate — appends raw records to the jsonl after every batch, so
#    partial progress always survives an interruption.
uv run python scripts/pearl_recalibration/generate_pearl_full.py

# 2. After a stall/interruption, pick up where it left off — only cells
#    with < --samples responses are regenerated.
uv run python scripts/pearl_recalibration/generate_pearl_full.py --resume

# 3. Clear error records and regenerate those cells.
uv run python scripts/pearl_recalibration/generate_pearl_full.py --retry-errors

# 4. Aggregate the raw jsonl into the final table (no API calls).
uv run python scripts/pearl_recalibration/generate_pearl_full.py --finalize-only
```

Options: `--samples N`, `--variant NAME` (default `protocol_fewshot`, the
frozen r13 prompt), `--temperature T` (default 0.3, the round-14 pilot winner),
`--batch-size N` (prompts per LLM batch, default 100), `--workers N` (default
50), and `--max-states N` (limit for smoke tests).

Raw records: `tables/pearl_12action/raw/results_full_<variant>.jsonl`, one
`{"state": {...}, "action": "...", "content|error": ...}` per line. Final table:
`tables/pearl_12action/pearl_bootstrap.json`.

### Why this design

- **Per-batch append (not per-chunk)**: the generator appends to the raw jsonl
  after each LLM batch returns. A stalled request therefore never risks losing
  completed work — everything already on disk survives, and `--resume` re-runs
  only what is missing. This is the same resilience Sprint 1 got from
  `request_helper.py --resume / --retry-errors` (see above).
- **Cell-level resume**: `--resume` counts only records with `content` (errors
  do not count), so a partially-filled cell is topped up to `--samples` rather
  than regenerated from scratch.
- **`--retry-errors`** rewrites the raw file without error records, then tops up
  the affected cells. Use it after a run finished with error lines.
- Errors are harmless to the table: `aggregate_to_table` (in
  `table_aggregate.py`) skips them and drops cells below its min-sample floor.

### Relation to the mini pilot

`generate_pearl_mini.py` is the 4-state pilot (temperature sweep, rounds 14-15)
that picked temperature 0.3 for the full run. `generate_pearl_full.py` uses the
same prompts, parser, and aggregator at full scale (1,404 cells, 14,040 calls).

## stable_maintainer: complete

The `stable_maintainer` persona run is complete: 22,320/22,320 prompts with 0
errors. The original 721 OpenRouter rate-limit errors were cleared using
OpenCode Zen with `request_helper --resume --persona=stable_maintainer --subdir=persona --provider=zen --concurrency=5`.
