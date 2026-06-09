# Subphase 1A: Config-Driven Data Layer

**Status:** `[ ]` Not started

**Dependencies:** None

**Parallelises with:** 1B, 1D (interface only), Dataset Exploration

---

## Gate Checklist

- [ ] Config schema can map arbitrary column names to semantic fields (steps, heart rate, sleep, etc.)
- [ ] Ingest pipeline reads a CSV/Parquet file from a config and produces a typed `Dataset` object
- [ ] Feature engineering pipeline runs from config (normalisation, aggregation, derived features)
- [ ] Synthetic data generator produces realistic-looking wearable data from configurable parameters
- [ ] `uv run pytest` passes for all 1A tests
- [ ] `uv run ruff check` and `uv run ty check` pass

---

## TDD Checklist

- [ ] Write test for config schema parsing (valid config parses, missing field rejects) *before* implementing parser
- [ ] Write test for ingest pipeline (correctly maps columns per config, rejects unknown column types gracefully) *before* implementing pipeline
- [ ] Write test for synthetic data shape + statistical properties (output shape matches params, step counts non-negative, no NaNs) *before* implementing generator

---

## Tech Notes

- **Ingest:** Polars lazy scan (`pl.scan_csv`, `pl.scan_parquet`). Config-driven column mapping applied before materialisation.
- **Feature engineering:** `FeaturePipeline.from_config(config)` builds a `pl.LazyFrame` computation graph (groupby, rolling windows, expressions). Materialised once to numpy arrays.
- **Simulation data:** `Dataset` is numpy-backed for fast random access. No DataFrame operations in the hot path.

## Key Interfaces

### `DataConfig`
```python
class DataConfig(BaseModel):
    file_path: str
    file_format: Literal["csv", "parquet"]
    column_mapping: dict[str, str]  # e.g. {"STEPS": "steps", "HR": "heart_rate"}
    feature_engineering: list[FeatureSpec]
```

### `FeaturePipeline`
```python
class FeaturePipeline:
    @staticmethod
    def from_config(config: DataConfig) -> FeaturePipeline

    def transform(self, raw: pl.LazyFrame) -> pl.LazyFrame
```

### `Dataset`
```python
@dataclass
class Dataset:
    user_ids: np.ndarray
    timestamps: np.ndarray
    features: dict[str, np.ndarray]
    metadata: dict[str, Any]
```

### `SyntheticDataGenerator`
```python
class SyntheticDataGenerator:
    def generate(self, n_users: int, n_weeks: int) -> Dataset
```

---

## Blocking Risks

- **Config format decisions:** YAML confirmed. No blocker.
- **Column mapping too rigid:** If a dataset has fundamentally different structure (e.g., per-minute vs daily aggregates), the ingest pipeline may need a code bump. Design the config to handle both resampled and raw formats.

---

## Reliability Patterns

Addresses the 7 design-level reliability gaps from audit #15 §3. Each gap
specifies: failure mode, mitigation, and a test sketch.

### Gap 1: Timeout on `pl.scan_csv` / `pl.scan_parquet`

**Failure mode:** Hung I/O blocks the entire experiment run. A network-mounted
Parquet file that times out at the OS level can stall for minutes before failing.

**Mitigation:**
- `polars_reader.scan_csv_with_timeout(path, timeout_s=30)` — wraps the
  `pl.scan_csv` call in a `concurrent.futures.ThreadPoolExecutor` with a
  30s timeout. On timeout, raises `TimeoutError` with the offending path.
- `polars_reader.scan_parquet_with_timeout(path, timeout_s=60)` — same
  pattern, 60s default.
- Defaults are overridable per-call. Both wrappers are the only entry point
  for raw file reads in 1A; direct `pl.scan_*` is banned.

**Test sketch:** `tests/unit/data/test_polars_reader.py::test_scan_csv_timeout`
— point at a fifo / hanging socket; verify `TimeoutError` after 30s.

### Gap 2: Lazy Polars defers errors to `collect()`

**Failure mode:** Corrupt file errors surface far from the read site. The
`pl.scan_csv` call succeeds; the error only appears when `collect()` runs
deep in the sim loop. The traceback points at `collect()`, not at the file
path.

**Mitigation:**
- Every `scan_*` call site is paired with eager validation at the read site:
  ```python
  lf = polars_reader.scan_csv_with_timeout(path, timeout_s=30)
  df = lf.collect()  # ← errors here, not deep in sim loop
  expected = DataConfig.expected_schema
  if df.schema != expected:
      raise SchemaMismatchError(path=path, expected=expected, actual=df.schema)
  ```
- The wrapper returns a materialised `pl.DataFrame`, not a `LazyFrame`,
  forcing the error to surface at the read site.

**Test sketch:** `tests/unit/data/test_polars_reader.py::test_lazy_error_at_read_site`
— corrupt CSV; verify `SchemaMismatchError` raised with the file path
in the error message, not a deep `pl.ComputeError` traceback.

### Gap 3: No `Dataset` shape validation

**Failure mode:** Misconfigured pipeline causes a deep `IndexError` in the
sim loop. The `Dataset` object looks fine at the boundary but has the wrong
shape (e.g., `features["steps"]` has shape `(100,)` when the MDP expects
`(100, 365)`).

**Mitigation:**
- `Dataset` is a `Protocol` with explicit shape invariants documented in
  `data/dataset.py`:
  - `user_ids.shape == (N,)`
  - `timestamps.shape == (N, T)`
  - `features[k].shape == (N, T)` for every key `k`
  - `metadata["n_users"] == N`, `metadata["n_timesteps"] == T`
- `Dataset.validate()` method raises `DatasetShapeError` with a clear message
  if any invariant is violated. Called at construction time, not lazily.
- 1B environment constructor calls `dataset.validate()` as its first line.

**Test sketch:** `tests/unit/data/test_dataset.py::test_shape_validation` —
construct a `Dataset` with mismatched `user_ids` and `features` shapes;
verify `DatasetShapeError` is raised with a message naming the offending
field.

### Gap 4: `REGISTRY` dict import-time failure

**Failure mode:** A bad implementation module (e.g., syntax error in
`rule_based.py`) kills the entire namespace. The `__init__.py` does
`from .rule_based import RuleBasedTransition` and a `SyntaxError` there
prevents any other component from being imported.

**Mitigation:**
- Concrete modules are imported inside a try/except in the `__init__.py`:
  ```python
  _registry: dict[str, type] = {}
  for _module in (_rule_based, _compound, _thompson_sampling):
      try:
          _module.register(_registry)
      except Exception as exc:
          logger.error("Failed to register %s: %s", _module.__name__, exc)
  ```
- A failed registration is logged at ERROR but does not prevent other
  components from registering. The runner's factory then surfaces a
  clear "X not registered" error when asked for the missing component.

**Test sketch:** `tests/unit/data/test_registry.py::test_import_time_failure_isolated`
— inject a module that raises on import; verify other modules still
register; verify the broken one is not in the registry.

### Gap 5: Shallow factory validation (`type in REGISTRY` only)

**Failure mode:** Cross-field mismatches slip through. A reward function
references a feature that doesn't exist; an agent's action space doesn't
match the environment's. The factory accepts the config and the error
surfaces at the first `step()` call.

**Mitigation:**
- Factory validation is a 3-layer check:
  - **Layer 1:** `type in REGISTRY` — type exists
  - **Layer 2:** Config schema validates against the type's expected
    fields (Pydantic `model_validate`)
  - **Layer 3:** A dummy step succeeds end-to-end: `env.reset()` →
    `agent.select_action()` → `env.step()` with no exception. The dummy
    step uses a 1-timestep episode with seed=0.
- All three layers must pass. Layer 3 is the integration test that catches
  cross-field mismatches.
- Failed validations raise `ConfigValidationError` with a message
  naming the failing layer and the offending field.

**Test sketch:** `tests/integration/test_dummy_step.py` — builds the
smallest valid config; runs the dummy step; verifies no exception.
A second test injects a bad config (reward references nonexistent
feature); verifies `ConfigValidationError` raised at Layer 3.

### Gap 6: `DataConfig.file_path` not resolved/validated

**Failure mode:** A relative path is resolved against the wrong directory
(e.g., the runner's CWD) and silently fails to find the file. The error
message is "file not found" but doesn't tell the user which directory
the path was resolved against.

**Mitigation:**
- `DataConfig.file_path` is resolved against a base path with explicit
  precedence:
  1. Explicit `base_path` field in the config (if set)
  2. `RL_HEALTH_DATA_ROOT` env var (if set)
  3. Fail-fast: raise `ConfigError` with a message naming all three
     candidates and instructing the user to set one.
- The resolved path is logged at INFO before any read attempt.
- Empty `file_path` is rejected at parse time (Pydantic `min_length=1`).

**Test sketch:** `tests/unit/data/test_data_config.py::test_path_resolution`
— three test cases: explicit `base_path`, `RL_HEALTH_DATA_ROOT` env var,
neither set (expect `ConfigError`).

### Gap 7: `week = t // 7` hardcoded in `StateView`

**Failure mode:** Wrong metadata for non-daily timesteps. The MDP
might be 5 decisions per day (HeartSteps design), so `t // 7` is wrong
by a factor of 5. Biases downstream research that uses `week` as a
feature.

**Mitigation:**
- `StateView` accepts a `timestep_duration: timedelta = timedelta(days=1)`
  field. The default is daily; override is config-driven.
- `week` is computed from the timestamp, not from `t`:
  ```python
  week = (timestamp - episode_start).days // 7
  ```
- The 1B environment constructor requires `timestep_duration` in the
  config (no silent default). 5x/day HeartSteps-style runs pass
  `timedelta(hours=24/5)`.
- An invariant is added: `timestep_duration` must be a positive
  `timedelta`. Zero or negative is rejected at parse time.

**Test sketch:** `tests/unit/transitions/test_state_view.py::test_week_from_timestamp`
— verify week calculation is correct for 5x/day timesteps. A second
test verifies zero/negative `timestep_duration` is rejected.

---

## Logging & Error Handling

See canonical setup in [`06 Code Design.md`](06%20Code%20Design.md#logging--error-handling-canonical).

Subphase-specific concerns for 1A (data layer):

- **Timeouts (CRITICAL):** All `pl.scan_csv` calls wrapped in a 30s timeout;
  all `pl.scan_parquet` calls wrapped in a 60s timeout. Implement as
  `polars_reader.scan_csv_with_timeout(path, timeout_s=30)`.
- **Lazy error surface:** Polars defers errors to `collect()`. Every `scan_*`
  call must be paired with eager validation: `df = scan.collect() ; assert
  df.schema == expected`. Failures are caught at the read site, not deep in the
  sim loop.
- **Validation failures:** Logged at ERROR with the offending file path and
  expected vs. actual schema. Skipped from the run; counted in
  `logs/failed_episodes.jsonl` (treated as ingest failures).
- **File-not-found:** Logged at ERROR with the resolved path (after
  `DataConfig.file_path` resolution against `RL_HEALTH_DATA_ROOT`). Fail-fast:
  cannot start the run without a valid path.
- **Synthetic fallback:** If real-data ingest fails 3 times consecutively,
  log WARNING and fall back to `SyntheticDataGenerator` with the seed from
  the run config. Never silently swap data sources.
- **DEBUG events:** schema of every loaded DataFrame, row count, time range
  covered, per-column null counts.
- **INFO events:** "Loaded dataset X (N users, M rows, T weeks) in 2.3s" once
  per ingest. Synthetic data generation: "Generated N synthetic users with
  seed S in 0.8s" once.

Related 1A tests:
- `tests/unit/data/test_polars_reader.py::test_scan_csv_timeout` — wrapped
  scan raises `TimeoutError` after 30s on a hung source.
- `tests/unit/data/test_polars_reader.py::test_lazy_error_at_read_site` —
  corrupt CSV raises on `collect()`, not on `scan()`.
- `tests/unit/data/test_dataset.py::test_schema_validation` — schema mismatch
  raises with a message naming the offending column.
- `tests/integration/data/test_synthetic_fallback.py` — 3 ingest failures
  triggers synthetic with the run seed.

---

## Public Datasets for Pipeline Testing

Phase 1 testing relies on three dataset categories: synthetic (generated on the
fly), WISDM, and ExtraSensory. All are open-access with no application required.

### WISDM (Wireless Sensor Data Mining)

- Kwapisz et al. (2011), ACM SIGKDD Explorations
- UCI ML repo URL: https://www.cis.fordham.edu/wisdm/dataset.php (Fordham University, Kwapisz's home institution — historically listed under UCI ML)
- 36 users, 6 activities (walking, jogging, stairs, sitting, standing, lying)
- Phone accelerometer, 20 Hz, ~1 GB
- Open access — no application needed
- Use: pipeline testing, label validation, synthetic distribution seeding

### ExtraSensory Dataset

- Vaizman et al. (2017), UbiComp
- http://extrasensory.ucsd.edu/
- 60 users, ~7 days each
- Phone sensors: accelerometer, gyroscope, compass, audio, location
- Labels: activity, mood, social context, phone usage
- Open access, ~6 GB
- Use: feature pipeline validation, multi-sensor ingestion testing

Phase 1 uses synthetic + WISDM/ExtraSensory for testing. HeartSteps V1/V2 data
comes in Phase 2 once data access requests are approved (see #16).
