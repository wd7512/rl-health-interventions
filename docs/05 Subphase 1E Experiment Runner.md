# Subphase 1E: Experiment Runner & Results

**Status:** `[ ]` Not started

**Dependencies:** 1C, 1D

**Parallelises with:** Nothing (final integration step)

---

## Gate Checklist

- [ ] CLI command `uv run rl-health-interventions --config experiment.yml` runs end-to-end
- [ ] Experiment config defines: which agents to compare, number of trials, environment config
- [ ] Results output: table comparing agent performance (mean reward, regret, engagement)
- [ ] Reproducibility: seeds and config snapshot saved with each run
- [ ] `uv run pytest` passes for all 1E tests
- [ ] `uv run ruff check` and `uv run ty check` pass

---

## TDD Checklist

- [ ] Write test for experiment config parsing *before* implementing runner
- [ ] Write test for reproducibility (same config + seed → same results) *before* implementing runner

---

## Key Interfaces

### `ExperimentConfig`
```python
class ExperimentConfig(BaseModel):
    data_config: DataConfig
    mdp_config: MDPConfig
    agents: list[AgentConfig]
    n_trials: int
    n_epochs: int
    seeds: list[int]
```

### `ExperimentResult`
```python
class ExperimentResult:
    config_snapshot: ExperimentConfig
    agent_results: dict[str, AgentResult]
```

---

## CLI Design

```
uv run rl-health-interventions --config experiments/demo.yml
```

Expected output:
- Console: summary table of agent performance
- `results/` directory: per-agent metrics, config snapshot, seed logs

---

## Blocking Risks

- **Integration complexity:** 1E depends on all subphases. If any upstream interface changes, 1E breaks. Mitigation: lock interfaces (1A Dataset, 1B Environment, 1D Agent) before 1E starts.
- **Comparison methodology:** What constitutes a "fair" comparison? Same environment config, same seeds, same evaluation protocol. Document this in the experiment config schema.
