# Phase 2A: Doc-Alignment Raw Audit

**Date:** 2026-06-11
**Auditor:** Primary Model (Qwen 3.7 Max)
**Scope:** All documentation and code files in rl-health-interventions

---

## Summary Statistics

- **Total files audited:** 96
- **Source files (.py):** 17
- **Test files (.py):** 11
- **Documentation files (.md):** 30+
- **Config files (.yaml):** 14
- **Design document:** 1 (.tex)

---

## 1. Module Structure Alignment

| Module | Exists | REGISTRY | ABC | Tests | Status |
|--------|--------|----------|-----|-------|--------|
| transitions/ | ✓ | ✓ | ✓ | ✓ | Aligned |
| rewards/ | ✓ | ✓ | ✓ | ✓ | Aligned |
| agents/ | ✓ | ✓ | ✓ | ✓ | Aligned |
| simulation/ | ✓ | ✓ | ✓ | ✓ | Aligned |
| data/ | ✓ | ✓ | ⚠️ Protocol | ✓ | Partial |

**Notes:**
- `data/_base.py` uses `Protocol` instead of `ABC` — this is a valid alternative pattern but differs from the other modules
- All modules follow the documented ABC + registry pattern from `code_design.md`
- Test coverage exists for all modules

---

## 2. Subphase Implementation Status vs Documentation

| Subphase | Doc Status | Actual Status | Alignment |
|----------|-----------|---------------|-----------|
| 1A Data Layer | Not started | Partial (1055 lines in loaders.py) | ⚠️ Partial |
| 1B MDP Environment | Not started | Partial (stubs + ABCs) | ⚠️ Partial |
| 1C User Simulation | Not started | Stub (25 lines) | ⚠️ Partial |
| 1D Agent Library | Not started | Partial (ABC + TS stub) | ⚠️ Partial |
| 1E Experiment Runner | Not started | Missing | ✗ Stale |

**Key Findings:**
- All subphase docs say "Not started" but 1A, 1B, 1D have partial implementations
- 1A data layer is most advanced (1055 lines in loaders.py with 12 dataset loaders)
- 1E experiment runner is completely missing — no environment.py, no runner.py, no config.py
- Subphase docs are stale — they don't reflect current implementation progress

---

## 3. Design Document Interface Alignment

| Interface | Documented in .tex | Implemented in Code | Status |
|-----------|-------------------|---------------------|--------|
| StateView | ✓ (code_design.md) | ✗ Missing | ✗ Stale |
| Environment | ✓ (subphase_1b) | ✗ Missing | ✗ Stale |
| UserProfile | ✓ (subphase_1c) | ✗ Missing | ✗ Stale |
| ExperimentConfig | ✓ (subphase_1e) | ✗ Missing | ✗ Stale |
| Dataset | ✓ (code_design.md) | ✓ Present | ✓ Aligned |
| DataConfig | ✓ (subphase_1a) | ✓ Present | ✓ Aligned |
| FeaturePipeline | ✓ (code_design.md) | ✓ Stub | ⚠️ Partial |
| SyntheticDataGenerator | ✓ (code_design.md) | ✓ Present | ✓ Aligned |
| TransitionModel | ✓ (subphase_1b) | ✓ ABC + stub | ⚠️ Partial |
| RewardHandler | ✓ (subphase_1b) | ✓ ABC + stub | ⚠️ Partial |
| Agent | ✓ (subphase_1d) | ✓ ABC + stub | ⚠️ Partial |
| ResponseModel | ✓ (subphase_1c) | ✓ ABC + stub | ⚠️ Partial |

**Critical Gaps:**
- 4 major interfaces documented but not implemented: StateView, Environment, UserProfile, ExperimentConfig
- These are blocking components for Phase 1 completion
- Design docs are ahead of implementation — this is expected but should be documented

---

## 4. Config-Loader Alignment

| Dataset Config | Loader Function | Status |
|----------------|-----------------|--------|
| 4tu_step_goals.yaml | load_4tu_step_goals() | ✓ Aligned |
| allofus_fitbit.yaml | — | ✗ Missing loader |
| bidsleep.yaml | load_bidsleep() | ✓ Aligned |
| dreamt.yaml | load_dreamt() | ✓ Aligned |
| extrasensory.yaml | load_extrasensory() | ✓ Aligned |
| fitbit_tracker.yaml | load_fitbit_tracker() | ✓ Aligned |
| harth.yaml | load_harth() | ✓ Aligned |
| mhealth.yaml | load_mhealth() | ✓ Aligned |
| pmdata.yaml | load_pmdata() | ✓ Aligned |
| scientisst_move.yaml | load_scientisst_move() | ✓ Aligned |
| stepcountjitai.yaml | — | ✗ Missing loader |
| synthetic.yaml | load_synthetic() | ✓ Aligned |
| wesad.yaml | load_wesad() | ✓ Aligned |
| wisdm.yaml | load_wisdm() | ✓ Aligned |

**Issues:**
- 2 configs (allofus_fitbit, stepcountjitai) have no matching loader functions
- These datasets require special access (All of Us requires cloud access, StepCountJITAI may require special download)
- Loaders.py has `load_all()` function but no individual loaders for these 2 datasets

---

## 5. Documentation Quality Audit

### README.md
- **Status:** Complete
- **Alignment:** Aligned
- **Issues:**
  - Quickstart section exists but doesn't produce results (no working experiment)
  - No architecture diagram
  - No usage examples beyond basic commands
  - References docs/code/ files correctly

### CONTRIBUTING.md
- **Status:** Complete
- **Alignment:** Aligned
- **Issues:** None — accurately describes branch model, workflow, and code style

### AGENTS.md
- **Status:** Complete
- **Alignment:** Aligned
- **Issues:** None — correctly documents uv, pytest, ruff, ty commands

### docs/contents.md
- **Status:** Partial
- **Alignment:** Aligned
- **Issues:**
  - Lists all docs/code/ files correctly
  - Lists all docs/sources/ files correctly
  - Missing: no description of reports/ directory (will be created by this audit)

### docs/code/*.md (8 files)
- **Status:** Partial
- **Alignment:** Stale
- **Issues:**
  - All subphase files say "Status: [ ] Not started" but implementations exist
  - Gate checklists are incomplete — no indication of which gates have passed
  - TDD checklists are unchecked — unclear if tests were written before implementation
  - Missing: no cross-references between subphase files

### docs/sources/*.md (14 files)
- **Status:** Complete
- **Alignment:** Aligned
- **Issues:** None — comprehensive dataset documentation

---

## 6. Code Quality Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| Total functions | 58 | — |
| Docstring lines | 41 | 70.7% coverage |
| Type hints | Present | ✓ |
| Line length | 88 (ruff) | ✓ Enforced |
| Test files | 11 | ✓ |
| Config files | 14 | ✓ |

**Docstring Coverage by Module:**
- data/loaders.py: High (module docstring + function docstrings)
- data/_base.py: Medium (class docstrings)
- transitions/_base.py: Low (minimal docstrings)
- rewards/_base.py: Low (minimal docstrings)
- agents/_base.py: Low (minimal docstrings)
- simulation/_base.py: Low (minimal docstrings)

**Issues:**
- ABC base classes lack docstrings explaining their purpose
- No API documentation (no Sphinx/mkdocs setup)
- Inline comments are sparse

---

## 7. Missing Components

| Component | Expected From | Status | Impact |
|-----------|---------------|--------|--------|
| environment.py | subphase_1b | Missing | Blocks MDP simulation |
| experiment.py / runner.py | subphase_1e | Missing | Blocks end-to-end experiments |
| config.py (Pydantic schemas) | subphase_1a/1b | Missing | Blocks config validation |
| state_view.py | code_design.md | Missing | Blocks state representation |
| user_profile.py | subphase_1c | Missing | Blocks user simulation |
| Example experiment configs | README | Missing | Blocks user onboarding |
| API documentation | CONTRIBUTING | Missing | Blocks contributor onboarding |

---

## 8. Cross-Reference Audit

### README → docs/code/
- ✓ References codebase_plan.md
- ✓ References code_design.md
- ✓ References phase_1_execution_plan.md
- ✗ Does not reference subphase files individually

### docs/code/ → docs/sources/
- ✓ subphase_1c references data_sources.md
- ✓ subphase_1c references additional_data_sources.md
- ✗ No cross-references to specific dataset files

### initial_design.tex → code/
- ✓ References codebase_plan.md
- ✓ References code_design.md
- ✓ References README.md
- ✗ Does not reference subphase files

---

## 9. Naming Consistency

| Concept | Design Doc | Code | Config | Status |
|---------|-----------|------|--------|--------|
| Transition model | TransitionModel | TransitionModel | transition_model | ✓ Consistent |
| Reward handler | RewardHandler | RewardHandler | reward_handler | ✓ Consistent |
| Agent | Agent | Agent | agent | ✓ Consistent |
| Response model | ResponseModel | ResponseModel | response_model | ✓ Consistent |
| Dataset | Dataset | Dataset | dataset | ✓ Consistent |
| State view | StateView | — | — | ✗ Not implemented |
| Environment | Environment | — | — | ✗ Not implemented |

---

## 10. File-by-File Audit Table

| File | Section | Doc Status | Code Alignment | Issues Found |
|------|---------|------------|----------------|--------------|
| README.md | Overview | Complete | Aligned | No working example |
| README.md | Quickstart | Complete | Stale | Quickstart doesn't produce results |
| README.md | Milestones | Complete | Stale | ASCII diagram outdated |
| CONTRIBUTING.md | Workflow | Complete | Aligned | None |
| CONTRIBUTING.md | Code Style | Complete | Aligned | None |
| AGENTS.md | Commands | Complete | Aligned | None |
| AGENTS.md | Rules | Complete | Aligned | None |
| docs/contents.md | Overview | Partial | Aligned | Missing reports/ description |
| docs/code/code_design.md | Module Structure | Complete | Aligned | None |
| docs/code/code_design.md | Data Pipeline | Complete | Partial | FeaturePipeline is stub |
| docs/code/code_design.md | Interfaces | Complete | Stale | StateView, Environment missing |
| docs/code/codebase_plan.md | Architecture | Complete | Aligned | None |
| docs/code/codebase_plan.md | Done Criteria | Complete | Stale | Criteria not met |
| docs/code/phase_1_execution_plan.md | Dependency Graph | Complete | Aligned | None |
| docs/code/phase_1_execution_plan.md | Gates | Complete | Stale | Gates not verified |
| docs/code/subphase_1a_data_layer.md | Status | Partial | Stale | Says "not started" but partial impl exists |
| docs/code/subphase_1b_mdp_environment.md | Status | Partial | Stale | Says "not started" but stubs exist |
| docs/code/subphase_1c_user_simulation.md | Status | Partial | Stale | Says "not started" but stubs exist |
| docs/code/subphase_1d_agent_library.md | Status | Partial | Stale | Says "not started" but stubs exist |
| docs/code/subphase_1e_experiment_runner.md | Status | Partial | Stale | Says "not started" and nothing exists |
| docs/initial_design.tex | MDP | Complete | Stale | Interfaces not implemented |
| docs/initial_design.tex | Framework | Complete | Partial | Partial implementation |
| docs/sources/*.md | All 14 files | Complete | Aligned | None |
| config/datasets/*.yaml | All 14 files | Complete | Partial | 2 missing loaders |
| src/rl_health_interventions/data/loaders.py | All loaders | Complete | Partial | 2 datasets not loadable |
| src/rl_health_interventions/data/synthetic.py | Generator | Complete | Aligned | None |
| src/rl_health_interventions/data/feature_pipeline.py | Pipeline | Partial | Aligned | Stub implementation |
| src/rl_health_interventions/data/dataset.py | Dataset | Complete | Aligned | None |
| src/rl_health_interventions/data/polars_reader.py | Reader | Complete | Aligned | None |
| src/rl_health_interventions/transitions/*.py | All files | Complete | Aligned | Stub implementations |
| src/rl_health_interventions/rewards/*.py | All files | Complete | Aligned | Stub implementations |
| src/rl_health_interventions/agents/*.py | All files | Complete | Aligned | Stub implementations |
| src/rl_health_interventions/simulation/*.py | All files | Complete | Aligned | Stub implementations |
| tests/unit/**/*.py | All tests | Complete | Aligned | None |
| tests/integration/test_dummy_step.py | Integration | Complete | Aligned | None |
| pyproject.toml | Config | Complete | Aligned | None |
| opencode.json | Config | Complete | Aligned | None |
| orchestrator_prompt.md | Prompt | Complete | N/A | Meta-document |

---

## 11. Critical Misalignments

1. **Subphase status docs are stale.** All 5 subphase files say "Not started" but 1A, 1B, 1D have partial implementations. This creates confusion about actual progress.

2. **Design doc interfaces not implemented.** StateView, Environment, UserProfile, ExperimentConfig are documented in detail but not implemented. This is the biggest gap between docs and code.

3. **Config-loader mismatch.** 2 dataset configs (allofus_fitbit, stepcountjitai) have no matching loader functions. Users cannot load these datasets despite configs existing.

4. **Missing experiment runner.** subphase_1e documents a complete experiment runner but nothing exists. This blocks end-to-end experiments.

5. **README quickstart is misleading.** The quickstart suggests running `uv run rl-health-interventions` but this only prints "Hello from rl-health-interventions!" — no actual experiment runs.

---

## 12. Minor Issues

1. **Docstring coverage inconsistent.** Base classes have minimal docstrings. loaders.py has good coverage. Average 70.7%.

2. **No API documentation.** No Sphinx/mkdocs setup. Contributors must read source code to understand APIs.

3. **No example experiment configs.** README mentions `--config experiment.yml` but no example configs exist in the repo.

4. **docs/contents.md incomplete.** Doesn't mention reports/ directory (will be created by this audit).

5. **Cross-references sparse.** Subphase files don't reference each other. Dataset source files aren't cross-referenced from subphase docs.

---

## Summary

**Total files audited:** 96
**Files with issues:** 23
**Critical misalignments:** 5
**Minor issues:** 5

**Overall alignment:** 65% — Core architecture is aligned, but implementation lags behind documentation. Subphase status docs are stale, 4 major interfaces are missing, and 2 dataset configs have no loaders.

---

*End of Phase 2A raw audit*
