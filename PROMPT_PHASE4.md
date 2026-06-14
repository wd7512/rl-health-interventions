# Tasks 4.1-4.2: Full Simulation Runner and Results

You are running the complete HeartSteps V2 simulation study and generating results.

The worktree is at /Users/williamdennis/rl-health-interventions-repro
The package is at paper_reproduction/

## Task 4.1: Full Simulation Runner

Create: paper_reproduction/simulation/run_study.py
Create: paper_reproduction/simulation/results.py

Wire together all components and run the complete simulation study.

The full simulation study:
1. Load/generate NHANES-like data
2. Partition into 3 folds for cross-validation
3. For each CV iteration:
   a. Construct prior from training batch
   b. Estimate noise variance from training batch
   c. Grid search for (gamma, w) using training batch
   d. Build generative models for test batch participants
   e. Run proposed algorithm 96 times per test participant
   f. Run TS Bandit 96 times per test participant
   g. Compute average total reward for each participant under each algorithm
4. Aggregate results across all 3 iterations
5. Compute improvement per participant: reward_proposed - reward_bandit
6. Report: number improved, average improvement, median improvement

The results module should:
- Store results in a structured format (dict or dataclass)
- Compute summary statistics
- Support reproducibility (seed management)
- Save results to JSON for later visualization

For initial implementation, use small numbers:
- 30 participants (10 per fold)
- 7 days per participant (extend to 90 via concatenation)
- 10 re-runs per participant (instead of 96)
- Small grid search (skip extreme values)

Tests:
- Full pipeline runs without errors
- Results contain expected keys
- Improvement is computable
- Results are deterministic with same seed

## Task 4.2: Results Visualization

Create: paper_reproduction/visualization/plot_results.py
Create: paper_reproduction/results/REPORT.md

Generate Figure 2 equivalent and summary table.

Figure 2 equivalent:
- Bar chart showing improvement per participant (sorted)
- X-axis: participant ID (sorted by improvement)
- Y-axis: improvement (reward_proposed - reward_bandit)
- Horizontal line at 0 (no improvement)
- Title: "Proposed Algorithm Improvement over TS Bandit"

Summary table:
- Number of participants improved / total
- Mean improvement
- Median improvement
- Standard deviation of improvement
- Min and max improvement

Also generate:
- Dose-response curves: intervention frequency over time for proposed vs TS Bandit
- Posterior mean traces for sample participants (showing how treatment effect estimates evolve)

Save all plots as PNG files in paper_reproduction/results/

The REPORT.md should contain:
- Summary of methodology
- Key results (numbers)
- Interpretation
- Comparison to paper's findings

Tests:
- Plots are generated without errors
- Report is readable
- Results match expected format

## Code Quality Requirements
- Follow existing code style
- Use logging (stdlib), never print()
- Type hints on all public methods
- Docstrings on all public classes and methods
- Run: uv run ruff check paper_reproduction/ tests/
- Run: uv run ruff format paper_reproduction/ tests/
- Run: uv run pytest tests/ -v
- All tests must pass

## DO NOT:
- Modify existing files
- Add new dependencies
- Use print() for output (use logging)
- Skip writing tests
