# Subphase 1E: Experiment Runner & Results

**Status:** `[ ]` Not started

**Dependencies:** 1C, 1D

**Parallelises with:** Nothing (final integration step)

---

## Gate Checklist

- [ ] CLI command `uv run rl-health-interventions --config experiment.yml` runs end-to-end
- [ ] Experiment config defines: which agents to compare, number of trials, environment config
- [ ] Results output: table comparing agent performance (mean reward, regret, adherence)
- [ ] Reproducibility: seeds and config snapshot saved with each run
- [ ] `uv run pytest` passes for all 1E tests
- [ ] `uv run ruff check` and `uv run ty check` pass

---

## TDD Checklist

- [ ] Write test for experiment config parsing (valid full experiment config loads, each sub-config delegates correctly) *before* implementing runner
- [ ] Write test for reproducibility (same config + seed → identical results across 2 runs) *before* implementing runner

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

### Factory / Experiment boundary

The `ExperimentFactory` constructs everything from config; the `Experiment`
executes and owns runtime state:

```python
class ExperimentFactory:
    @staticmethod
    def build(config: ExperimentConfig) -> Experiment

class Experiment:
    def run(self) -> ExperimentResult
```

The factory is stateless — same config always gives the same wiring. The
experiment owns episode counters, accumulators, checkpoints, and the
results output.

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

---

## Logging & Error Handling

See canonical setup in [`code-design.md`](code-design.md#logging--error-handling-canonical).

Subphase-specific concerns for 1E (experiment runner):

- **CLI flags (owner: 1E):** `--verbose`, `--quiet`, `--log-file` are
  implemented in the runner's argparse setup, not in the subphase
  modules. Forwarded to the root logger.
- **Per-episode exception isolation:** The `for episode in range(N):` loop
  wraps the entire body in `try/except Exception`. Caught exceptions:
  - Log at ERROR with `episode_id`, `agent_id`, `user_id`, exception
    type, message, and traceback
  - Write to `logs/failed_episodes.jsonl` (append-only, one JSON object
    per failed episode)
  - Increment a `failed_episode_count` field on the run
  - Continue to the next episode
- **Progress heartbeat:** Every 100 episodes (configurable via
  `runner.progress_interval`), log INFO with: episodes completed,
  episodes failed, mean reward so far, elapsed time, ETA based on
  current throughput.
- **Structured trace emission:** At episode end, append one JSON line
  to `logs/episodes.jsonl` with the canonical schema (see 06 Code
  Design.md). This is the primary data product for downstream analysis.
- **Checkpoint:** Every 500 episodes (configurable), pickle the agent
  state to `checkpoints/run_<id>_ep<NNNN>.pkl`. On restart, the runner
  detects the latest checkpoint and resumes from there.
- **Run summary:** At run end, log INFO with: total episodes,
  successful/failed, wall-clock time, mean reward, std reward, config
  hash, git SHA.

Related 1E tests:
- `tests/integration/test_episode_isolation.py` — one bad episode does
  not stop the run; the run completes with the right failure count.
- `tests/integration/test_heartbeat.py` — heartbeat emitted at the
  configured interval.
- `tests/integration/test_checkpoint_resume.py` — kill the runner at
  episode 750, restart, verifies resume from episode 750.
- `tests/integration/test_structured_trace.py` — `logs/episodes.jsonl`
  has one line per successful episode with all required fields.
