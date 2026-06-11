# Phase 5, Domain 5: CI/CD & Developer Tooling Audit

**Date:** 2026-06-11
**Auditor:** Primary Model (Qwen 3.7 Max)

---

## CI/CD Pipeline

**GitHub Actions workflows:**
- No .github/workflows/ directory found
- No CI/CD pipeline configured
- Tests are not automatically run on push/PR

**Status:** ✗ No CI/CD

---

## Pre-commit Hooks

**Status:** ✗ Not configured.

- No .pre-commit-config.yaml
- No pre-commit hooks installed
- Developers must manually run `uv run ruff format`, `uv run ruff check`, `uv run ty check`, `uv run pytest`

---

## Dependency Management

**Tool:** uv (modern Python package manager)
**Lock file:** uv.lock present ✓
**Dependency specification:** pyproject.toml with version ranges

**Dependencies (runtime):**
- datasets>=5.0.0
- huggingface-hub>=1.18.0
- kagglehub>=1.0.2
- numpy>=1.24.0
- pandas>=3.0.3
- polars>=1.0.0
- pydantic>=2.0.0
- ucimlrepo>=0.0.7
- wfdb>=4.3.1

**Dependencies (dev):**
- pytest>=8.4.0,<10
- ruff>=0.14.10,<0.15
- ty>=0.0.42,<0.1

**Vulnerability scan:** No obvious known-vulnerable versions. All dependencies are recent (2024-2026 releases).

**Build reproducibility:** ✓ uv.lock ensures reproducible installs.

---

## Linting/Formatting

**Tools configured:**
- ruff (formatting + linting) — line-length 88, target Python 3.11
- ty (type checking) — configured in pyproject.toml

**Enforcement:** ✗ Not enforced. No CI, no pre-commit hooks. Developers must run manually.

**Configuration:**
```toml
[tool.ruff]
line-length = 88
target-version = "py311"
```

---

## Build Reproducibility

**Can the environment be reproduced from scratch using only tracked files?**

**Yes.** ✓

- pyproject.toml defines all dependencies
- uv.lock pins exact versions
- Python version specified: >=3.11
- Build system: uv_build

**Steps:**
1. Install uv
2. `uv sync --dev`
3. Environment reproduced

**Missing:** No Docker/container specification for full environment reproducibility (OS-level deps, system packages).

---

## Missing Tooling

### Critical
1. **CI/CD pipeline** — no automated testing on push/PR
2. **Pre-commit hooks** — no automatic linting/formatting before commit

### High
3. **Docker/container** — no containerised environment for reproducibility
4. **Makefile or task runner** — no unified command for common tasks (test, lint, format, build)

### Medium
5. **Dependency vulnerability scanning** — no automated security checks
6. **Coverage reporting** — no pytest-cov or coverage tracking
7. **Documentation hosting** — no ReadTheDocs/GitHub Pages setup

---

## Developer Experience Friction Points

1. **Manual command execution** — developers must remember to run ruff, ty, pytest before pushing
2. **No unified task runner** — no `make test` or `make lint` command
3. **No CI feedback** — PRs merge without automated quality checks
4. **No coverage visibility** — developers don't know test coverage percentage
5. **No dependency update automation** — no Dependabot/Renovate for security updates

---

## Summary

**CI/CD status:** None
**Pre-commit hooks:** None
**Dependency management:** Good (uv + lock file)
**Linting/formatting:** Configured but not enforced
**Build reproducibility:** Good (uv.lock)
**Top tooling gap:** No CI/CD pipeline — tests don't run automatically

---

*End of Domain 5 audit*
