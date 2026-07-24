# pearl_random results

Golden fixtures produced by:

```bash
uv run python docs/experimental_phases/pearl_random/run_experiments.py \
  --all --seeds 50 --output docs/experimental_phases/pearl_random/results \
  --json --confirm-overwrite
```

The JSON fixtures are used by `tests/regression/test_pearl_random.py`.
