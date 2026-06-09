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
