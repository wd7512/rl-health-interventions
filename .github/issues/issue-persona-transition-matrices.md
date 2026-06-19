# Issue 3: Design 4 persona hand-curated transition matrices

**Labels:** `enhancement`, `phase-1`, `external-input`
**Assignees:** @wd7512

## Context

Mengyan asked: *"how'd you plan to build fixed transition matrices from the 4 personas? can we write it out"*

Current approach: a single transition matrix shared by all users. For heterogeneous simulation, we need distinct matrices per persona. This is the precursor task for M-04 (User Simulation Engine) but narrower in scope — pure design + config, no code changes to the simulation module.

The 4 archetypes are defined in `initial_design.tex §3` and `docs/plans/subphase_1c_user_simulation.md`.

## Specification

### The 4 archetypes (from design doc)

| Archetype | Behaviour | Response to interventions | Burden accumulation |
|-----------|-----------|--------------------------|-------------------|
| **Goal-driven** | Responds to reminders and goal feedback | High response to `goal_reminder`, `progress_feedback` | Moderate |
| **Social responder** | Responds to motivational prompts | High response to `motivational_prompt` | Moderate |
| **Resistant** | Low response overall | Flat/negative response to all interventions | Fast (2× rate) |
| **Stable maintainer** | Already active | Low marginal gain from any action | Slow |

### Hand-curated matrices

For each archetype, produce a `P(s' | s, a)` matrix. With the 2-state `{sedentary, active}` and 6-action space from Issue #1, each matrix is 6×2×2.

**Example — Goal-driven archetype:**

| Current | Action | → Active | → Sedentary | Rationale |
|---------|--------|----------|-------------|-----------|
| sedentary | no_message | 0.10 | 0.90 | Baseline without intervention |
| sedentary | motivational_prompt | 0.25 | 0.75 | Moderate response |
| sedentary | walking_suggestion | 0.30 | 0.70 | Higher than generic |
| sedentary | goal_reminder | 0.45 | 0.55 | **Highest** — goal-driven responds to goals |
| sedentary | recovery_suggestion | 0.15 | 0.85 | Low — not relevant to this archetype |
| sedentary | progress_feedback | 0.40 | 0.60 | **High** — feedback motivates |
| active | no_message | 0.70 | 0.30 | Maintains activity |
| active | motivational_prompt | 0.72 | 0.28 | Mild boost |
| active | walking_suggestion | 0.68 | 0.32 | Mild reactance |
| active | goal_reminder | 0.65 | 0.35 | Slight reactance (already active) |
| active | recovery_suggestion | 0.73 | 0.27 | Perceived as helpful |
| active | progress_feedback | 0.70 | 0.30 | Neutral — already knows progress |

**All matrices must satisfy:**
- Reactance theory: interventions help sedentary users but can mildly backfire for already-active users
- Burden penalty: frequent interventions degrade response probability over time (documented, not embedded in matrix)
- Sum to 1.0 per `(s, a)` row
- All values in [0, 1], valid probability distributions

### Deliverables

1. **Design doc section** — Write up the 4 archetype definitions and their transition matrices in `docs/design/initial_design.tex §3` (or new appendix)
   - For each archetype: narrative description, transition matrix table, rationale for key probability choices
   - Document citations: StepCountJITAI for response magnitude ranges, HeartSteps for burden/engagement decay
   - Document limitations: discrete archetypes are a simplification; Phase 2 should use continuous parameters

2. **Config YAMLs** — Create 4 config files (or a single multi-profile config):
   - `config/archetype_goal_driven.yaml`
   - `config/archetype_social_responder.yaml`
   - `config/archetype_resistant.yaml`
   - `config/archetype_stable_maintainer.yaml`

3. **Validation script** — Verify matrices sum to 1.0, match Reactance theory expectations, produce distinct behaviour distributions (ANOVA on continuous summary metrics of simulated trajectories, e.g. mean active ratio per episode)

4. **Results doc** — Run Thompson Sampling on each archetype config for 10 seeds, produce a `.tex` + `.pdf` in `docs/mvp/extensions.tex` showing per-archetype agent performance (similar pattern to `docs/mvp/mvp_specification.tex`)

## Out of scope
- Burden accumulation/decay dynamics in the Environment (requires M-04 code changes)
- Continuous parameter sampling across archetypes (Phase 2)
- Real-data calibration (requires HeartSteps/TILES access — Phase 2)

## Related
- `docs/design/initial_design.tex §3` (archetype definitions)
- `docs/plans/subphase_1c_user_simulation.md`
- `docs/plans/ROADMAP.md` (M-04: User Simulation Engine)
- Issue #1 (6-action space — matrices depend on full action set)
- `docs/mvp/mvp_specification.tex` (reference for result doc format)
