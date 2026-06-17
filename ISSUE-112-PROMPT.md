Implement contextual Thompson Sampling for the rl-health-interventions project.

## GitHub Issue
https://github.com/wd7512/rl-health-interventions/issues/112

## What to build
Extend the existing ThompsonSamplingAgent to support contextual action selection. When `contextual: true` in config, the agent maintains separate `(alpha, beta)` posteriors for each `(context, action)` pair instead of global per-action posteriors.

## Context feature
`activity` (sedentary/active) — already in `StateView`. This gives 2 contexts × 2 actions = 4 posterior pairs.

## Files to modify
1. `src/rl_health_interventions/agents/thompson_sampling.py` — extend class
2. `src/rl_health_interventions/config/schemas.py` — add optional `contextual` and `context_feature` fields to `AgentConfig`
3. `tests/unit/agents/test_thompson_sampling.py` — add tests

## Config
`config/contextual_thompson.yaml` already exists in the repo root with:
```yaml
agents:
  - type: thompson_sampling
    alpha_prior: 1
    beta_prior: 1
    contextual: true
    context_feature: activity
```

## Implementation details
- When `contextual=False` (default): existing behaviour, no changes
- When `contextual=True`: 
  - `posteriors` becomes `dict[tuple[str, str], Posterior]` keyed by `(context_value, action)`
  - `select_action` extracts `state.activity` and samples from relevant posteriors
  - `update` extracts `state.activity` and updates the correct `(context, action)` posterior
- Backward compatible: existing configs without `contextual` field work identically

## Tests to add
1. Contextual TS with context-dependent rewards — verify it outperforms plain TS when contexts have different optimal actions
2. Contextual TS with uniform rewards — verify it converges to same behaviour as plain TS
3. Config without `contextual` field — verify no regression
4. Verify `AgentConfig` accepts the new optional fields

## Verification
```bash
cd repos/.worktrees/contextual-bandits
uv run ruff check src tests
uv run ruff format .
uv run pytest tests/unit/agents/test_thompson_sampling.py -v
uv run pytest tests/ -q  # full suite must pass
```
