# Phase 2B: Orchestrator Synthesis

**Source:** reports/phase2_doc_alignment_raw.md
**Analysed by:** Primary Model (Qwen 3.7 Max)
**Date:** 2026-06-11

---

## 1. Issue Categorisation

### Critical Misalignment (3 issues)

1. **Experiment runner absent.** `subphase_1e_experiment_runner.md` documents a complete ExperimentFactory + Experiment + CLI pipeline. None of this exists in code. The README references `uv run rl-health-interventions --config experiment.yml` which currently prints "Hello" and exits. A user following the README will believe the system works when it does not.

2. **StateView and Environment missing.** `code_design.md` defines StateView as the bridge between Dataset and the simulation loop, and `subphase_1b` defines Environment with step/reset API. Both are central to the MDP execution. Without them, no simulation can run. The design docs describe these in detail (dataclass definitions, from_dataset classmethod, step return types) but code has only the ABC stubs that return `state` unchanged.

3. **Config validation absent.** `code_design.md` specifies "3-layer validation" (Pydantic schema → registry lookup → dummy step). `subphase_1a` defines DataConfig with Pydantic. The actual code has DataConfig in `data/_base.py` but no ExperimentConfig, no MDPConfig, no validation layers. The config-first architecture claim is unsubstantiated — configs exist as YAML files but nothing reads them.

### Stale Docs (5 issues)

4. **All 5 subphase status fields say "Not started"** when data layer (1055 lines), loaders (12 datasets), and ABC+registry scaffolding are implemented. This is the most pervasive stale-doc issue — it affects every subphase file.

5. **README milestone diagram shows unchecked boxes (□)** for all Phase 1 items. In reality, the data layer, registry pattern, and basic tests are done. The diagram misrepresents progress.

6. **codebase_plan.md "Done Criteria" lists 5 deliverables** — none are met. The CLI exists but doesn't run experiments. Config schema is partially documented. README exists but doesn't enable running experiments.

7. **phase_1_execution_plan.md shows Dataset Exploration as ✅ DONE** but marks all other subphases as `[ ]`. The data loader implementation (PR #73) is not reflected.

8. **Decision Log in initial_design.tex** lists "Is MDP formalisation reasonable? — Awaiting supervisor approval" as Open. If supervisor has approved (the code implements the MDP structure), this should be updated.

### Missing Docs (4 issues)

9. **No API documentation.** No Sphinx/mkdocs setup. No docstrings on ABC base classes. Contributors must reverse-engineer interfaces from code.

10. **No example experiment configs.** README mentions `--config experiment.yml` but no example YAML exists in the repo. Users cannot learn the config schema from examples.

11. **No architecture diagram in README.** code_design.md describes the pipeline but README has no visual overview. New contributors cannot understand the system at a glance.

12. **No changelog or release notes.** No CHANGELOG.md. No versioning strategy documented. pyproject.toml says 0.1.0 but no indication of what changed between versions.

### Inconsistent Naming (2 issues)

13. **data/_base.py uses Protocol, other modules use ABC.** This is a valid pattern choice but inconsistent with the documented "ABC + registry" pattern. code_design.md says "_base.py contains only the ABC" but data/_base.py uses Protocol.

14. **Config key naming vs Python class naming.** code_design.md says "Keys are explicit snake_case aliases, not class names." This is correctly implemented but the mapping is not documented anywhere — a user must read __init__.py files to discover available keys.

### Minor (3 issues)

15. **docs/contents.md doesn't mention reports/ directory.** Will be partially resolved by this audit.

16. **Cross-references between subphase files are sparse.** subphase_1c depends on 1A and 1B but doesn't link to their files.

17. **orchestrator_prompt.md exists at repo root.** This is a meta-document for AI agents. It should either be in docs/ or excluded from the repo (it's an operational artifact, not project documentation).

---

## 2. TOP 5 Most Dangerous Doc–Code Divergences

### #1: README Quickstart Promises a Working System That Doesn't Exist

**What's wrong:** README says `uv run rl-health-interventions` runs the system. It actually prints "Hello from rl-health-interventions!" and exits.

**Why it's dangerous:** A researcher cloning this repo (e.g., Swapnil's team at NUS) will believe the framework is functional. They will attempt to configure experiments, discover nothing works, and lose confidence in the entire project. This is the first thing external stakeholders see.

**Incorrect behaviour it could cause:** A collaborator might build analysis pipelines assuming the experiment runner exists, only to discover mid-integration that the core loop is missing. This wastes weeks of work.

### #2: Subphase Status Fields Systematically Overstate Progress

**What's wrong:** All 5 subphase files say "Status: [ ] Not started" when significant implementation exists.

**Why it's dangerous:** This creates two failure modes simultaneously: (a) the team thinks nothing is done and may duplicate work, and (b) the gate checklists are unreliable — if status is wrong, the gate criteria may also be wrong. A subphase could be marked "gate passed" when critical components are still missing.

**Incorrect behaviour it could cause:** A developer starting subphase 1C (user simulation) would read "Dependencies: 1A, 1B" and assume neither is done, when in fact the data layer and MDP stubs exist and could be built upon. This leads to either redundant reimplementation or incorrect assumptions about what's available.

### #3: StateView Interface Defined in Docs But Not in Code

**What's wrong:** code_design.md defines StateView as a dataclass with features dict, user_id, timestamp, metadata, and a from_dataset classmethod. No such class exists in code.

**Why it's dangerous:** StateView is the central data structure that flows through the entire simulation loop. Every component (Environment, Agent, ResponseModel) takes StateView as input. Without it, the interfaces defined in subphase_1b through 1e are all using `Any` type annotations, which means:
- Type checking provides no safety
- Interface contracts are implicit
- Components cannot be composed without runtime errors

**Incorrect behaviour it could cause:** A developer implementing the Agent might define `select_action(self, state: dict)` while the Environment passes a `StateView` object. The type checker won't catch this because both use `Any`. The error surfaces only at runtime, potentially after hours of simulation.

### #4: Config-First Architecture Claimed But Not Implemented

**What's wrong:** The entire design philosophy is "config-first" — YAML configs drive all components. But no code reads YAML configs. DataConfig exists as a Pydantic model but nothing instantiates it from YAML. ExperimentConfig doesn't exist at all.

**Why it's dangerous:** The config-first claim is the project's primary value proposition. If configs don't actually drive the system, the architecture is a standard hardcoded framework with unused YAML files. This undermines the entire design rationale and wastes the effort spent on 14 dataset config files.

**Incorrect behaviour it could cause:** A researcher might spend time crafting YAML configs for their dataset, only to discover the framework ignores them. Or worse, they might assume the configs work, run experiments with misconfigured parameters, and get silently incorrect results.

### #5: Two Dataset Configs Have No Loaders

**What's wrong:** allofus_fitbit.yaml and stepcountjitai.yaml exist but no corresponding loader functions exist in loaders.py.

**Why it's dangerous:** These are two of the most important datasets for the project. StepCountJITAI is the primary reference for the MDP design (cited in initial_design.tex). All of Us Fitbit is a planned Phase 2 validation dataset. Having configs without loaders creates a false sense of completeness — someone checking "do we support this dataset?" will see the config and assume yes.

**Incorrect behaviour it could cause:** A Phase 2 developer might attempt to validate against StepCountJITAI data, discover no loader exists, and need to implement it from scratch — duplicating the feasibility study already done in docs/sources/stepcount_jitai_dataset.md.

---

## 3. Waterfall Drift Patterns

Evidence that docs were written before code architecture was settled:

1. **code_design.md defines a 3-layer validation system** (Pydantic → registry → dummy step) but code has zero validation layers. The design was aspirational — the implementer built the registry pattern but never added validation. This is classic waterfall: design the ideal system, implement what's easy, leave the hard parts for "later."

2. **subphase_1a defines DataConfig with file_path, file_format, column_mapping, feature_engineering** — but the actual DataConfig in data/_base.py only has file_path, file_format, column_mapping, base_path. The feature_engineering field was dropped during implementation, likely because FeaturePipeline is still a stub. The docs describe the planned interface; the code reflects what was actually built.

3. **code_design.md describes a Dataset → StateView bridge** with a from_dataset classmethod. The Dataset dataclass exists but StateView doesn't. The implementer built the data storage (Dataset) but not the access pattern (StateView). The docs assumed both would be built together; the code built them separately and stopped halfway.

4. **initial_design.tex references Gymnasium-style interfaces** (step/reset) but pyproject.toml explicitly excludes Gymnasium ("no Gymnasium, no SB3" in Decision Log). The design doc was written with Gymnasium in mind, then the decision was made to avoid it, but the .tex was never updated to reflect this.

5. **subphase_1e defines ExperimentResult with config_snapshot and agent_results** but no experiment runner exists to produce results. The output format was designed before the input pipeline was built. Classic waterfall: design the output before building the system that produces it.

---

## 4. Doc Gaps Mapped to Roadmap Milestones

| Doc Gap | Blocks Milestone | Why |
|---------|-----------------|-----|
| StateView interface undefined in code | M-02 (MDP Environment) | Environment.step() needs StateView as input/output type |
| ExperimentConfig schema missing | M-05 (Experiment Runner) | Cannot wire components without config schema |
| Config YAML loading not implemented | M-01 (Data Layer), M-05 (Runner) | Config-first architecture requires YAML → Pydantic pipeline |
| Subphase status fields stale | M-01 through M-05 | Cannot track progress if status is wrong |
| No example experiment configs | M-05 (Runner), M-06 (Validation) | Users cannot learn config schema without examples |
| API documentation absent | M-07 (Documentation) | Contributors cannot understand interfaces |
| Environment class missing | M-02 (MDP Environment) | Core simulation loop cannot run |
| UserProfile missing | M-03 (User Simulation) | User archetypes cannot be configured |
| 2 dataset configs without loaders | M-01 (Data Layer) | allofus_fitbit and stepcountjitai not loadable |
| README quickstart misleading | M-07 (Documentation) | First impression for external stakeholders |

---

## 5. Doc Debt Score

**Formula:** (number of fully aligned sections / total sections audited) × 100

**Audit results:**
- Total sections audited: 37 (from file-by-file table in raw report)
- Fully aligned (Complete + Aligned): 24
- Partial or stale: 13

**Doc Debt Score: 65% — 24/37 sections fully aligned.**

This is acceptable for an early-stage project but will deteriorate rapidly if implementation continues without doc updates. The 13 misaligned sections are concentrated in the most critical areas (subphase status, interface definitions, config validation).

---

## 6. Doc Remediation Tickets

### Ticket 1: Update all subphase status fields and gate checklists

**What is wrong:** All 5 subphase files in docs/code/ say "Status: [ ] Not started" when implementations exist for 1A (partial), 1B (partial), and 1D (partial). Gate checklists are unchecked even where gates may have passed.

**What correct state looks like:** Each subphase file accurately reflects implementation status:
- 1A: Status: [~] In progress — data layer partially implemented (loaders, Dataset, synthetic generator done; FeaturePipeline stub; config YAML loading missing)
- 1B: Status: [~] In progress — ABC stubs done; Environment, StateView, TransitionModel implementations missing
- 1C: Status: [ ] Not started — accurate (only ABC stub exists)
- 1D: Status: [~] In progress — Agent ABC + ThompsonSampling stub done; real TS implementation missing
- 1E: Status: [ ] Not started — accurate (nothing exists)

Gate checklists should have individual items checked where implemented.

**PR acceptance criteria:**
- [ ] All 5 subphase files have accurate status fields
- [ ] Gate checklist items reflect actual implementation state
- [ ] README milestone diagram updated to match
- [ ] phase_1_execution_plan.md dependency graph updated

### Ticket 2: Implement or document missing core interfaces

**What is wrong:** StateView, Environment, UserProfile, and ExperimentConfig are defined in design docs but not in code. All component interfaces use `Any` type annotations, defeating type safety.

**What correct state looks like:** Either (a) implement all 4 interfaces as documented, or (b) update design docs to reflect that these interfaces are deferred and document the rationale. If deferred, all dependent subphase docs should note the dependency explicitly.

**PR acceptance criteria:**
- [ ] StateView dataclass implemented per code_design.md spec OR design doc updated with deferral rationale
- [ ] Environment class with step/reset implemented OR design doc updated
- [ ] UserProfile schema implemented OR design doc updated
- [ ] ExperimentConfig schema implemented OR design doc updated
- [ ] All ABC base classes updated from `Any` to concrete types once interfaces are defined
- [ ] `uv run ty check` passes with concrete types

### Ticket 3: Add example experiment config and fix README quickstart

**What is wrong:** README references `uv run rl-health-interventions --config experiment.yml` but no example config exists. The quickstart runs a hello-world script, not an experiment. External stakeholders cannot learn how to use the system.

**What correct state looks like:** README quickstart either (a) runs a minimal working experiment with an example config, or (b) explicitly states "Phase 1 implementation in progress — see docs/code/ for architecture" and points to the design docs. An example experiment config (experiments/demo.yml) demonstrates the intended config schema.

**PR acceptance criteria:**
- [ ] experiments/demo.yml exists with documented fields
- [ ] README quickstart either runs a working experiment OR clearly states implementation status
- [ ] Config schema documented (either in README, docs/, or as docstrings on Pydantic models)
- [ ] A new contributor can understand what the system will do when complete

---

## Quality Gate 2 Verification

- [x] Raw report covers every file in the repo (96 files audited)
- [x] Orchestrator analysis has all 5 numbered components
- [x] Doc Debt Score calculated: 65% (24/37 sections aligned)
- [x] All TOP 5 dangerous divergences have explicit "why dangerous" explanations
- [x] 3 Critical misalignments documented (experiment runner, StateView/Environment, config validation)

**Gate 2: PASSED**

---

*End of Phase 2B orchestrator synthesis*
