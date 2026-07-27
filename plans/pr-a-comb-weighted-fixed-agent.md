# PR-A: Com-B Weighted Fixed Agent

**Status:** Ready for implementation  
**Depends on:** None  
**Risk:** Low  
**Estimated effort:** 4-6 hours

## Overview

Implement `ComBWeightedFixedAgent` — a COM-B theory-based fixed policy agent that reproduces PEARL's Fixed arm logic. This agent selects nudge themes based on barrier-score weighted multinomial sampling and delivery timing based on participant preference.

## Design Decisions

### Architecture

- **Extends `Agent` directly** (not `FixedAgent`) — `FixedAgent`'s contract is "constant action," which violates the stochastic nature of COM-B sampling
- **Hardcoded vocabulary** — 6 COM-B themes × 2 timings as class constants, matching PEARL's fixed design
- **No fallback logic** — construction-time validation ensures all 12 combinations exist; if validation passes, fallback is dead code

### Action Space

- **12 COM-B actions** (6 themes × 2 timings): `ability_morning`, `ability_afternoon`, `perceived_benefit_morning`, etc.
- **`idle` excluded** from COM-B agent's selection set — `idle` is only for the Control arm
- **Validation at construction** — all 12 combinations must be present in the `actions` list; extras (including `idle`) are allowed
- **Default action list** — if `actions=None`, generate all 12 combinations from hardcoded themes × timings

### Score Loading

- **Dual input modes** — accept either `inline_comb_scores` (dict) OR (`persona_comb_file` + `persona_name`), but not both
- **Mutual exclusion enforced** — error if both provided, error if neither provided
- **File path resolution** — relative to current working directory (experiment run location)
- **`time_preference` in JSON** — lives alongside COM-B scores per persona, matching PEARL's survey design (each participant had both COM-B scores and timing preference)

### Barrier Computation

- **Formula** — `barrier = 5 - likert_score` per theme
- **Zero-barrier exclusion** — themes with Likert=5 (barrier=0) are excluded from multinomial sampling
- **All-zero error** — if all barriers are zero (all Likert=5), raise `ValueError` at construction time
- **Normalization** — barriers normalized to probabilities for multinomial sampling

### Timing Logic

- **Three valid preferences** — `morning`, `afternoon`, `no_preference`
- **Default to `no_preference`** — if not specified in JSON or config, use 50/50 split
- **70/30 split** — if preference stated, 70% preferred time, 30% other
- **50/50 split** — if `no_preference`, equal probability

### Random Number Generation

- **Consistent with framework** — accept `seed: int = 42`, create `self._rng = np.random.default_rng(seed)`
- **Dedicated Generator** — isolated from other components, ensures reproducibility

### Configuration Schema

- **New agent type** — `"comb_weighted_fixed"` added to `_KNOWN_AGENT_TYPES`
- **Required fields** — either (`persona_comb_file` + `persona_name`) OR `inline_comb_scores`
- **Optional override** — `time_preference` can be provided in config to override JSON value
- **Reject learning params** — `alpha_prior`, `beta_prior`, `epsilon`, `action`, `contextual` all rejected

### Synthetic Personas

- **5 archetypes** — base, goal_driven, social_responder, stable_maintainer, resistant
- **Hand-crafted** — represent tails and middle of COM-B score distribution
- **Documented as synthetic** — not claiming to reproduce PEARL's participant data
- **Cover behavioral space** — neutral, high motivation, low motivation, high social need, moderate

## Implementation Plan

### Step 1: Define Class Constants and Constructor

```python
class ComBWeightedFixedAgent(Agent):
    THEMES = frozenset({
        "ability", "perceived_benefit", "planning", 
        "prioritization", "social_opportunity", "physical_opportunity"
    })
    TIMINGS = frozenset({"morning", "afternoon"})
    
    def __init__(
        self,
        actions: list[str] | None = None,
        seed: int = 42,
        comb_scores: dict[str, int] | None = None,
        persona_comb_file: str | None = None,
        persona_name: str | None = None,
        time_preference: str | None = None,
    ) -> None:
        # Load scores (file or inline, mutual exclusion)
        # Compute barriers (5 - likert), exclude zeros
        # Validate all-zero barriers → error
        # Validate actions (all 12 combinations present)
        # Create RNG
```

### Step 2: Implement Score Loading

- If `comb_scores` provided → use directly
- Else if `persona_comb_file` + `persona_name` → load JSON, extract persona
- Else → error (neither provided)
- If both → error (mutual exclusion)
- Extract `time_preference` from JSON if not provided as parameter
- Validate scores are in range [1, 5]

### Step 3: Implement Barrier Computation

- `barriers = {theme: max(0, 5 - score) for theme, score in scores.items()}`
- Filter out zero-barrier themes
- If no themes remain → error
- Normalize to probabilities for multinomial

### Step 4: Implement Action Validation

- Generate expected 12 combinations: `{f"{theme}_{timing}" for theme in THEMES for timing in TIMINGS}`
- If `actions is None` → use all 12
- Else → check all 12 are in `actions` (extras OK, including `idle`)
- If any missing → error listing all missing actions

### Step 5: Implement `select_action`

- Sample theme from multinomial with barrier weights
- Sample timing based on preference (70/30 or 50/50)
- Return `f"{theme}_{timing}"`

### Step 6: Implement `update` and `on_day_end` as no-ops

### Step 7: Update `schemas.py`

- Add `"comb_weighted_fixed"` to `_KNOWN_AGENT_TYPES`
- Add fields: `persona_comb_file`, `persona_name`, `inline_comb_scores`, `time_preference`
- Add validation: mutual exclusion, required fields, time_preference enum

### Step 8: Create `config/pearl/comb_scores.json`

5 personas with COM-B scores and time preferences:

```json
{
  "base": {
    "ability": 3, "perceived_benefit": 2, "physical_opportunity": 4,
    "planning": 2, "prioritization": 3, "social_opportunity": 3,
    "time_preference": "morning"
  },
  "goal_driven": {
    "ability": 4, "perceived_benefit": 4, "physical_opportunity": 3,
    "planning": 3, "prioritization": 4, "social_opportunity": 2,
    "time_preference": "afternoon"
  },
  "social_responder": {
    "ability": 3, "perceived_benefit": 3, "physical_opportunity": 2,
    "planning": 2, "prioritization": 2, "social_opportunity": 5,
    "time_preference": "morning"
  },
  "stable_maintainer": {
    "ability": 4, "perceived_benefit": 3, "physical_opportunity": 3,
    "planning": 3, "prioritization": 3, "social_opportunity": 2,
    "time_preference": "no_preference"
  },
  "resistant": {
    "ability": 2, "perceived_benefit": 1, "physical_opportunity": 1,
    "planning": 1, "prioritization": 1, "social_opportunity": 1,
    "time_preference": "no_preference"
  }
}
```

### Step 9: Register Agent

- Add `register()` function in `fixed.py` to register `"comb_weighted_fixed"`

### Step 10: Write Comprehensive Tests

Follow the test plan in the "Test Coverage" section below.

## Test Coverage

### `TestComBWeightedFixedAgent` — Agent Logic

**Theme sampling:**
- Single dominant barrier (one theme has barrier=4, rest=0) → 100% that theme
- Equal barriers (all themes score=1, barrier=4 each) → uniform distribution across 6 themes
- One zero-barrier theme excluded (social_opportunity=5) → never sampled, other 5 themes share probability
- Two zero-barrier themes excluded → 4 themes remain
- Distribution matches weights over 10,000 samples (chi-squared or tolerance-based)
- All-zero barriers raises `ValueError` at construction

**Timing selection:**
- `morning` preference → ~70% morning, ~30% afternoon over 10,000 samples
- `afternoon` preference → ~30% morning, ~70% afternoon
- `no_preference` → ~50/50
- Default (no time_preference in JSON) → `no_preference` behavior

**Action string construction:**
- Returns `"{theme}_{timing}"` format
- All 12 possible actions produced over many samples
- Action always in the provided `actions` list

**Seed reproducibility:**
- Same seed → identical 1,000-action sequence
- Different seeds → different sequences (with high probability)

**Update / on_day_end:**
- `update()` is no-op (doesn't change distribution)
- `on_day_end()` is no-op
- State argument ignored (like FixedAgent)

**Actions parameter:**
- `actions=None` → defaults to all 12 COM-B combinations
- `actions` with all 12 + `idle` → works, only selects from 12
- `actions` missing one COM-B combination → raises `ValueError` at construction
- `actions` missing multiple → error lists all missing
- Empty `actions` list → raises `ValueError`

### `TestBarrierComputation`

- Likert=1 → barrier=4
- Likert=5 → barrier=0 (excluded)
- Likert=3 → barrier=2
- All Likert=5 → all barriers=0 → error
- Barrier scores sum correctly for multinomial normalization
- Negative Likert scores (if allowed) → barrier > 4

### `TestScoreLoading`

**File-based:**
- Valid JSON file + persona name → loads correctly
- Missing file → `FileNotFoundError`
- Missing persona name in JSON → `KeyError` or `ValueError`
- JSON with missing theme scores → error
- JSON with out-of-range scores (<1 or >5) → error or warning

**Inline scores:**
- Valid dict → loads correctly
- Empty dict → error (all-zero barriers)
- Missing theme → error

**Mutual exclusion:**
- Both file and inline → `ValueError`
- Neither file nor inline → `ValueError`

### `TestComBWeightedFixedConfig` — Schema Validation

- `persona_comb_file` + `persona_name` → valid
- `inline_comb_scores` only → valid
- Both → `ValueError`
- Neither → `ValueError`
- `time_preference="morning"` → valid
- `time_preference="afternoon"` → valid
- `time_preference="no_preference"` → valid
- `time_preference="night"` → `ValueError`
- `time_preference=""` → `ValueError`
- Rejects `alpha_prior`, `beta_prior`, `epsilon`, `action`, `contextual`
- `persona_comb_file=""` → rejected
- `persona_name=""` → rejected

### `TestRegistry` (additions to existing file)

- `"comb_weighted_fixed" in REGISTRY`
- `make("comb_weighted_fixed", comb_scores={...}, actions=[...])` returns `ComBWeightedFixedAgent`
- Parametrized kwargs test

### `TestIntegration`

- Full episode with `ComBWeightedFixedAgent` in a 12-action config (if config exists) or synthetic config
- All actions in trajectory are valid COM-B combinations
- Action distribution over episode matches expected barrier weights

## Files to Change

### Core Implementation
- `src/rl_health_interventions/agents/fixed.py` — add `ComBWeightedFixedAgent` class
- `src/rl_health_interventions/config/schemas.py` — add validation for `"comb_weighted_fixed"`
- `config/pearl/comb_scores.json` — 5 synthetic personas

### Tests
- `tests/unit/agents/test_fixed_agent.py` — comprehensive test suite
- `tests/unit/agents/test_agents_registry.py` — add registry tests

## Verification Steps

1. **Unit tests pass**: `uv run pytest tests/unit/agents/test_fixed_agent.py -v`
2. **Registry test passes**: `uv run pytest tests/unit/agents/test_agents_registry.py -v`
3. **Type check passes**: `uv run ty check --exclude tests/`
4. **Lint passes**: `uv run ruff check`
5. **Format check passes**: `uv run ruff format --check .`
6. **Manual smoke test**: Instantiate agent with inline scores, verify action distribution

## Out of Scope

- Experiment configs (PR-C)
- Transition models (PR-B)
- Posterior burden mechanism (PR-D)
- Documentation updates (PR-E)

## References

- PEARL paper: Lee et al. (2025). "A Personalized Exercise Assistant using Reinforcement Learning (PEARL)." arXiv:2508.10060.
- Deep analysis: `docs/research/recreations/pearl-rct-2025/pearl-deep-analysis.md`
- Phase 2 plan: `docs/plans/phase-2-pearl-matched-config.v1.md`
- Atomic decomposition: `plans/pr-266-pearl-atomic-decomposition.v2.md`
