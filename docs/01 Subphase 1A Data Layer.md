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

- [ ] Write test for config schema parsing *before* implementing parser
- [ ] Write test for ingest pipeline *before* implementing pipeline
- [ ] Write test for synthetic data shape + statistical properties *before* implementing generator

---

## Key Interfaces

### `ConfigSchema`
```python
class DataConfig(BaseModel):
    column_mapping: dict[str, str]  # e.g. {"STEPS": "steps", "HR": "heart_rate"}
    feature_engineering: list[FeatureSpec]
```

### `Dataset`
```python
class Dataset:
    users: list[UserID]
    timestamps: pd.DatetimeIndex
    features: dict[str, np.ndarray]  # semantic field → array
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
