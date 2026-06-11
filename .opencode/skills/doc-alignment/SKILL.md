---
name: doc-alignment
description: Use when checking that derived Markdown documentation aligns with a single source-of-truth document. Deploys parallel subagents to audit small file batches and produce a structured discrepancy report.
---

# Doc Alignment

Use this skill to audit documentation for consistency. A single source-of-truth file
(usually `.tex`, `.md`, or a spec) is compared against all `.md` documentation files
in the repository.

## Workflow

1. **Get the source-of-truth filename.** Ask the user if not already provided.
2. **Read it in full** — you need the complete picture before comparing.
3. **Discover derived docs.** Glob `**/*.md` excluding `.opencode/`, node_modules,
   `.venv`, and any other non-docs directories.
4. **Batch the files.** Split into batches of **≤4 files each**. Ideally 1 file per
   subagent for deepest analysis. Group related files together when batching
   (e.g., all subphase docs together).
5. **Launch parallel subagents** (use `task` tool, `subagent_type: general`).
   Each agent receives:
   - The full source-of-truth text
   - Its batch of file paths
   - The discrepancy categories below
   - The output format below
   - Instruction to read every file in its batch, compare each against the source
     of truth, and return structured results
6. **Compile results.** Deduplicate, sort by severity (HIGH > MEDIUM > LOW),
   present to the user before making changes.

## Discrepancy Categories

| Category | What to flag | Severity |
|---|---|---|
| **Contradiction** | Directly conflicting statement (e.g., `[0,1]` vs `ℤ≥₀` for same variable) | HIGH |
| **Omission** | Concept/decision present in source but entirely absent from derived doc | HIGH |
| **Numeric** | Different formula, count, range, or constant | MEDIUM |
| **Structural** | Different module/component breakdown, dependency order | MEDIUM |
| **Terminological** | Different name for same concept (`UserSim` vs `simulation/`) | LOW |
| **Extra content** | Derived doc says something source doesn't cover at all | LOW |

## Output Format

Each subagent returns a list of entries:

```
- file: path/to/file.md:line_number
  severity: HIGH | MEDIUM | LOW
  category: contradiction | omission | numeric | structural | terminological | extra
  description: One-sentence summary of the mismatch
  source: Exact quote from source of truth
  doc: Exact quote from the derived doc
```

## Rules

- Source of truth always wins — resolve by updating the derived doc.
- Do not modify the source-of-truth file.
- Report everything; let the user triage.
- If a batch has 0 files, skip that subagent.
- Read every file entirely — don't rely on grep alone.
- If only 1–2 derived docs exist, skip delegation and compare directly.
