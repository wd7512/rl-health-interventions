# Code Design

**Guiding philosophy:** *A Philosophy of Software Design* (Ousterhout).
Deep modules, information hiding, strategic thinking, define errors out of existence.

---

## Module Structure

Each pluggable concept gets a flat package with a `_base.py` ABC and a
`REGISTRY` dict in `__init__.py`:

```
transitions/               ← pluggable transition models
    _base.py               ← TransitionModel ABC
    rule_based.py          ← RuleBasedTransition
    __init__.py            ← imports + REGISTRY dict
rewards/                   ← pluggable reward functions
    _base.py               ← RewardHandler ABC
    compound.py            ← CompoundReward
    __init__.py            ← imports + REGISTRY dict
agents/                    ← pluggable agents
    _base.py               ← Agent ABC
    thompson_sampling.py   ← ThompsonSamplingAgent
    __init__.py            ← imports + REGISTRY dict
simulation/                ← pluggable user response models
    _base.py               ← ResponseModel ABC
    rule_based.py          ← RuleBasedResponse
    __init__.py            ← imports + REGISTRY dict
data/                      ← data layer
    _base.py               ← IngestPipeline ABC
    feature_pipeline.py    ← Config-driven feature engineering on Polars
    polars_reader.py       ← Reads CSV/Parquet via Polars lazy API
    dataset.py             ← Dataset (numpy arrays, for sim loop)
    synthetic.py           ← SyntheticDataGenerator
    __init__.py            ← imports + REGISTRY dict
```

Rules:
- `_base.py` contains only the ABC. No implementations.
- Each implementation is its own file, named after the class in snake_case.
- `__init__.py` explicitly imports every implementation and populates a
  `REGISTRY` dict:
  ```python
  from .rule_based import RuleBasedTransition

  REGISTRY: dict[str, type[TransitionModel]] = {
      "rule_based": RuleBasedTransition,
  }
  ```
- No auto-discovery. The `__init__.py` is the manifest.

---

## Data Pipeline

### Design

```
Raw files (CSV, Parquet, ...)      ← researcher provides these
    ↓
Polars LazyFrame scan               ← config-driven schema mapping
    ↓
FeaturePipeline.from_config(config) ← config-driven transforms (agg, window, diff, etc.)
    ↓  (Polars lazy graph, materialised once)
numpy arrays                         ← materialised for sim loop
    ↓
Dataset (user × time index)          ← random access for StateView slices
```

Polars owns the feature engineering stage. Its lazy API maps 1:1 to the
config-driven spec — grouping, rolling windows, expressions. The framework
builds a `pl.LazyFrame` from the config and materialises it once. The result
is converted to numpy arrays for the simulation loop, which needs fast random
access by user and time rather than query capability.

After materialisation, the simulation depends only on numpy — no Polars
dependency in the hot path.

### Why Polars (not Pandas, not SQL)

| Need | Why Polars wins |
|------|----------------|
| Config-driven expressions | Lazy API lets us build computation graphs from config without executing until materialise |
| Groupby + window aggs | `.group_by_dynamic()`, `.rolling()` match wearable data structure (per-user time series) |
| File format support | CSV, Parquet, IPC natively |
| Memory efficiency | Streaming, zero-copy, arrow-backed |
| Dependency weight | ~30MB — worth it for what it saves us building |

### Dataset

```python
@dataclass
class Dataset:
    user_ids: np.ndarray          # shape (n_users,)
    timestamps: np.ndarray         # shape (n_users, n_timesteps)
    features: dict[str, np.ndarray]  # shape (n_users, n_timesteps) each
    metadata: dict[str, Any]       # config snapshot, column mappings, etc.
```

Materialised from a Polars lazy pipeline. The sim loop indexes into these
arrays to build `StateView` objects. No DataFrame operations in the hot path.

### Dataset → StateView Bridge

The simulation loop never receives a full `Dataset`. It receives per-step
`StateView` slices built from the materialised feature arrays:

```python
@dataclass
class StateView:
    features: dict[str, float]     # single-timestep feature values
    user_id: int
    timestamp: int
    metadata: dict[str, Any]       # week number, body measure flag, etc.

    @classmethod
    def from_dataset(cls, dataset: Dataset, user_idx: int, t: int) -> StateView:
        return cls(
            features={name: array[user_idx, t] for name, array in dataset.features.items()},
            user_id=user_idx,
            timestamp=dataset.timestamps[user_idx, t],
            metadata={"week": t // 7},
        )
```

The bridge is explicit: the `Experiment` loop indexes `Dataset` by user and
time step, calls `StateView.from_dataset()`, and passes the result to the
environment and agent. This is a single function, not a new component —
no separate bridge module needed.

### Action Type

Throughout the framework, actions are `int` indices into the action space
defined in config. The config defines the label-to-index mapping. Components
never use opaque `Action` objects.

A single factory that takes the full `ExperimentConfig` and returns a ready-to-run
experiment:

```python
class ExperimentFactory:
    @staticmethod
    def build(config: ExperimentConfig) -> Experiment
```

Responsibilities:
1. Validate the full config tree via Pydantic (layer 1)
2. Instantiate each component from the config using `REGISTRY` lookups (layer 2)
3. Wire components together
4. Run one dummy `step()` to catch wiring errors before the main loop (layer 3)
5. Return an `Experiment` object with a `run()` method

The factory is the single entry point. Configuration, instantiation, and wiring
never leak outside it.

### Adding a new component

A researcher adds a custom transition model:

1. Create `transitions/my_custom_model.py` with a class inheriting `TransitionModel`
2. In `transitions/__init__.py`, add:
   ```python
   from .my_custom_model import MyCustomTransition
   REGISTRY["my_custom_model"] = MyCustomTransition
   ```
3. Reference it in config:
   ```yaml
   transition_model: my_custom_model
   ```

No other framework code changes. The factory discovers it via the registry dict.

---

## Dataset Config Schema

Dataset configs are YAML files stored in `config/datasets/`. Each config
describes a single dataset's source, schema, and column roles.

```yaml
name: wisdm
source_url: https://www.cis.fordham.edu/wisdm/dataset.php
license: public
file_format: csv  # or parquet
expected_columns:
  user_id: string
  timestamp: datetime
  activity: string  # target
  accelerometer_x: float
  accelerometer_y: float
  accelerometer_z: float
user_id_column: user_id
timestamp_column: timestamp
feature_columns: [accelerometer_x, accelerometer_y, accelerometer_z]
target_column: activity
```

Required fields:

- `name` — short identifier for the dataset
- `source_url` — where the raw file can be obtained
- `license` — license type (e.g. `public`, `cc-by-4.0`)
- `file_format` — `csv` or `parquet`
- `expected_columns` — map of column name → dtype
- `user_id_column` — column to use as user identifier
- `timestamp_column` — column to use as timestamp
- `feature_columns` — list of column names used as model inputs
- `target_column` — column name for the label / target variable

Configs are static files read by the config parser. No PyYAML dependency is
needed at runtime unless dynamic loading is added later.

---

## Validation

### Layer 1: Schema Validation (Pydantic)

The full `ExperimentConfig` is a Pydantic `BaseModel`. Every field has types,
defaults, and descriptions. Loading a config file automatically validates types,
required fields, and value ranges. A malformed config fails with a precise error
message at load time.

### Layer 2: Component Compatibility

After parsing, the factory checks that the requested components are compatible:

```python
def _validate_compatibility(self, config: ExperimentConfig) -> None:
    if config.mdp_config.reward.type not in rewards.REGISTRY:
        raise ConfigError(f"Unknown reward type: {config.mdp_config.reward.type}")
```

Cross-component checks ensure references are valid (e.g. reward function
uses only state variables that exist, fatigue threshold is positive).

### Layer 3: Dummy Step

After building, the factory calls a single step with a minimal state:

```python
env = self._build_environment(config)
state = env.reset()
next_state, reward, done = env.step(state, first_action)
```

This catches wiring errors, shape mismatches, and runtime contract violations
before the main experiment loop starts.

---

## Interfaces

### `TransitionModel`

```python
class TransitionModel(ABC):
    @abstractmethod
    def transition(self, state: StateView, action: int, profile: UserProfile) -> StateView: ...
```

### `RewardHandler`

```python
class RewardHandler(ABC):
    @abstractmethod
    def reward(self, state: StateView, action: int, profile: UserProfile) -> tuple[float, bool]: ...
```

### `Agent`

```python
class Agent(ABC):
    @abstractmethod
    def select_action(self, state: StateView) -> int: ...
    def update(self, state: StateView, action: int, reward: float, next_state: StateView) -> None: ...
```

### `ResponseModel`

```python
class ResponseModel(ABC):
    @abstractmethod
    def response(self, state: StateView, action: int, profile: UserProfile) -> StateView: ...
```

### `StateView`

Not an ABC. A lightweight, immutable data class (dataclass or Pydantic) that
wraps the raw feature arrays and exposes named fields. All framework components
receive and return state through `StateView`, not raw dicts or arrays.

---

## Dependencies

Minimal. The strict dependencies for Phase 1:

- **Pydantic** — config validation
- **numpy** — numerical operations (standard for RL)
- **polars** — config-driven feature engineering, lazy streaming, file I/O
- **pytest** — dev dependency

No Gymnasium. No SB3. No PyTorch (Phase 2 for deep agents).

---

## Testing

```
tests/
    unit/
        transitions/       ← each transition model in isolation
        rewards/           ← each reward function in isolation
        agents/            ← each agent in isolation
        simulation/        ← each response model in isolation
        factory/           ← factory validation and wiring
    integration/
        test_dummy_step.py ← factory builds and step succeeds
        test_config.yml    ← minimal config is loadable
    end_to_end/
        test_experiment.py ← full pipeline: config → build → train → results
```

Rules:
- Every ABC method has at least one test proving it works with a real implementation.
- Every subphase gate requires all tests in that subphase's modules to pass.
- Integration tests mirror the researcher's workflow: write config, run, inspect output.
- Test organisation maps to modules, not subphase numbers:
  `tests/unit/transitions/` covers both 1B's transition gate and 1C's response model gate.
- The factory's Layer 3 dummy step has a dedicated integration test
  (`tests/integration/test_dummy_step.py`).
- Plugin discovery (`REGISTRY` dict) has a test ensuring every registered
  component can be instantiated from a minimal valid config.

---

## Logging & Error Handling (canonical)

The framework's observability contract. Every subphase doc references this
section; subphase docs only specify what's unique to that subphase.

### Logger namespace

- Every module: `logger = logging.getLogger(__name__)`
- Top-level runner: `rl_health_interventions.runner`
- Subphase namespaces:
  - `rl_health_interventions.data`
  - `rl_health_interventions.transitions`
  - `rl_health_interventions.rewards`
  - `rl_health_interventions.agents`
  - `rl_health_interventions.simulation`
  - `rl_health_interventions.experiment`

### Log levels

| Level | When |
|-------|------|
| DEBUG | Per-step trace (state, action, reward, next_state). Disabled by default. |
| INFO | Episode boundary, config load, checkpoint, sweep start/end. |
| WARNING | Retry, fallback to synthetic, missing optional config. |
| ERROR | Failed episode (caught, logged, written to DLQ). |
| CRITICAL | Unrecoverable — config invalid, cannot start. |

### Log format

- Console: `%(asctime)s %(levelname)s [%(name)s] %(message)s`
- File (episode-level traces): JSON, one object per line, written via stdlib
  `logging.handlers.WatchedFileHandler` with a `JsonFormatter` (custom, ~30 lines
  in `rl_health_interventions/logging.py`).

### Episode-level structured trace

Every episode emits a JSON line to `logs/episodes.jsonl` with:

```json
{
  "episode_id": "uuid4",
  "config_hash": "sha256",
  "total_reward": 0.87,
  "steps": 365,
  "agent_id": "thompson_sampling",
  "user_id": "synthetic_042",
  "timestamp": "2026-06-09T14:30:00Z",
  "seed": 42
}
```

### CLI flags (in experiment runner)

- `--verbose` → DEBUG
- `--quiet` → WARNING
- `--log-file <path>` → default `logs/<run_id>.log`

### Exception handling (for 1000+ episode runs)

- Per-episode try/except wraps the entire `for episode in range(N):` body
- Caught exceptions: log at ERROR, write episode context to
  `logs/failed_episodes.jsonl`, continue
- One bad episode must not lose all progress

### Progress heartbeat

- Log every N episodes (configurable, default 100): elapsed time, ETA, mean reward
  so far. Implementation: counter in runner, no external dependency.

### Retry / circuit breaker for I/O

- Data ingestion (Polars `scan_csv` / `scan_parquet`): explicit timeout
  via `pl.Config.set_engine_affinity` + wrapper, default 30s for CSV, 60s for
  Parquet
- HTTP fetches: configurable timeout (default 60s), exponential backoff
  (3 retries, 1s / 2s / 4s)
- Circuit breaker: 5 consecutive I/O failures → fail fast with a clear error
  message naming the failing path
