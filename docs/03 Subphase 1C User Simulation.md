# Subphase 1C: User Simulation Engine (Rule-Based)

**Status:** `[ ]` Not started

**Dependencies:** 1A, 1B, Dataset Exploration

**Gate requires:** Dataset exploration report completed AND rule-based profiles working

**Parallelises with:** 1D (after 1A/1B gates clear)

---

## Gate Checklist

- [ ] Configurable user profiles instantiated from config (goal-driven, social responder, resistant, stable maintainer)
- [ ] Behavioural response model produces plausible step-count changes given state + action
- [ ] Backlash/fatigue mechanism implemented (burden threshold, response decay)
- [ ] Dataset exploration report evaluating All of Us and UK Biobank for simulator training exists in docs/
- [ ] `uv run pytest` passes for all 1C tests
- [ ] `uv run ruff check` and `uv run ty check` pass

---

## TDD Checklist

- [ ] Write test for each behavioural archetype producing expected response direction *before* implementing models
- [ ] Write test for fatigue accumulation and decay *before* implementing burden logic

---

## Key Interfaces

### `UserProfile`
```python
class UserProfile(BaseModel):
    archetype: Literal["goal_driven", "social_responder", "resistant", "stable_maintainer"]
    parameters: dict[str, float]
```

### `ResponseModel`
```python
class ResponseModel(ABC):
    @abstractmethod
    def response(self, state: State, action: Action, profile: UserProfile) -> float
```

---

## Dataset Exploration

Investigating two public datasets for future simulator calibration:

| Dataset | Source | Size | Access |
|---------|--------|------|--------|
| All of Us Fitbit | Nature Medicine 2026 | 59K participants, 14yr | Researcher workbench |
| UK Biobank Accelerometer | npj Digital Medicine 2024 | 100K participants, 7 days ea. | Application required |

Exploration report should cover:
- Feature availability (steps, HR, sleep, etc.)
- Population diversity
- Licensing / access barriers
- Feasibility of extracting behavioural response patterns

---

## Blocking Risks

- **Real data may not arrive:** Without real data calibration, the simulation remains synthetic-only. Gate only requires the *report*, not confirmed access.
- **Archetype validity:** The 4 archetypes are theoretical. No guarantee they match real user behaviour. Document this as an explicit assumption.
