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
