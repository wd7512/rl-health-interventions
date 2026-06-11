# Phase 5, Domain 4: Data & Configuration Audit

**Date:** 2026-06-11
**Auditor:** Primary Model (Qwen 3.7 Max)

---

## Config File Inventory

| Config File | Purpose | Required Fields | Documented Defaults | Env-Var Coverage |
|-------------|---------|-----------------|---------------------|------------------|
| config/datasets/synthetic.yaml | Synthetic data generation params | name, generation params | ✓ | ✗ |
| config/datasets/4tu_step_goals.yaml | 4TU coaching dataset schema | expected_columns, feature_columns | ✓ | ✗ |
| config/datasets/allofus_fitbit.yaml | All of Us Fitbit dataset schema | expected_columns, feature_columns | ✓ | ✗ |
| config/datasets/bidsleep.yaml | BidSleep sleep staging schema | expected_columns, feature_columns | ✓ | ✗ |
| config/datasets/dreamt.yaml | DREAMT dataset schema | expected_columns, feature_columns | ✓ | ✗ |
| config/datasets/extrasensory.yaml | ExtraSensory dataset schema | expected_columns, feature_columns | ✓ | ✗ |
| config/datasets/fitbit_tracker.yaml | Fitbit fitness tracker schema | expected_columns, feature_columns | ✓ | ✗ |
| config/datasets/harth.yaml | HARTH dataset schema | expected_columns, feature_columns | ✓ | ✗ |
| config/datasets/mhealth.yaml | mHealth dataset schema | expected_columns, feature_columns | ✓ | ✗ |
| config/datasets/pmdata.yaml | PMData multi-source schema | expected_columns, feature_columns | ✓ | ✗ |
| config/datasets/scientisst_move.yaml | Scientisst Move schema | expected_columns, feature_columns | ✓ | ✗ |
| config/datasets/stepcountjitai.yaml | StepCountJITAI schema | expected_columns, feature_columns | ✓ | ✗ |
| config/datasets/wesad.yaml | WESAD dataset schema | expected_columns, feature_columns | ✓ | ✗ |
| config/datasets/wisdm.yaml | WISDM dataset schema | expected_columns, feature_columns | ✓ | ✗ |
| pyproject.toml | Project metadata and dependencies | name, version, dependencies | ✓ | N/A |
| opencode.json | OpenCode AI agent configuration | model, instructions, skills | ✓ | N/A |

---

## .env.example or Equivalent

**Status:** ✗ Missing.

No .env.example file exists. Environment variables used in code:
- `RL_HEALTH_DATA_PATH` — data directory override (data/_base.py)
- `DATA_PATH` — alternative data directory (data/_base.py)
- `KAGGLE_API_TOKEN` — Kaggle authentication (data/loaders.py)
- `PHYSIONET_USERNAME` — PhysioNet authentication (data/loaders.py)
- `PHYSIONET_PASSWORD` — PhysioNet authentication (data/loaders.py)
- `WFDB_USERNAME` — WFDB authentication (data/loaders.py)
- `WFDB_PASSWORD` — WFDB authentication (data/loaders.py)

These are documented in code but not in a central .env.example file.

---

## Schema Documentation

**Status:** ⚠️ Partial.

- Dataset config schemas are self-documenting (YAML field names are descriptive)
- No JSON Schema or Pydantic validation for dataset configs
- DataConfig in data/_base.py defines the Python schema but doesn't validate YAML configs
- No documentation of what each YAML field means or valid value ranges

---

## Secrets Management

**Status:** ✓ Good.

- No hardcoded tokens, API keys, or credentials in source code
- Kaggle credentials read from environment variable or ~/.kaggle/kaggle.json
- PhysioNet credentials read from environment variables
- Authentication checks are graceful (log warning, return None if missing)

---

## Reproducibility from Cold Start

**Can someone clone the repo and reproduce an experiment following only the documented config?**

**No.**

Blockers:
1. No example experiment config exists
2. Config YAML files define dataset schemas but nothing reads them
3. No MDPConfig, AgentConfig, or ExperimentConfig schemas defined
4. No config loading pipeline (YAML → Pydantic)
5. `uv run rl-health-interventions` doesn't run an experiment

**Steps to reproduce (current state):**
1. Clone repo
2. `uv sync --dev`
3. `uv run pytest` — tests pass
4. `uv run rl-health-interventions` — prints "Hello" and exits
5. No experiment runs

---

## Health Data Handling

**Status:** ✗ Not documented.

- No PHI/PII data flow documentation
- No data anonymisation procedures documented
- No consent tracking mechanism
- No data access controls or audit logging
- No GDPR/HIPAA compliance documentation

Phase 1 uses synthetic data only, so this is not an immediate blocker. However, Phase 2 datasets (All of Us, UK Biobank, HeartSteps) require strict data handling documentation.

---

## Summary

**Config files:** 16 (14 dataset configs + pyproject.toml + opencode.json)
**Secrets issues:** None (good)
**Cold-start verdict:** Cannot reproduce experiment from cold start
**Top gaps:** No .env.example, no config validation, no experiment config schema, no health data handling docs

---

*End of Domain 4 audit*
