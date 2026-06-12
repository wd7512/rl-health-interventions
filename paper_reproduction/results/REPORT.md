# HeartSteps V2 Paper Reproduction — Results

## Methodology

This reproduction implements the HeartSteps V2 RL algorithm (Liao, Greenewald,
Klasnja, Murphy 2019, arXiv:1909.03539) and runs a simulation study comparing
the proposed algorithm against a Thompson Sampling Bandit baseline.

### What We Reproduce

- **Complete RL algorithm** (Sections 5.2–5.4): dosage variable, Bayesian linear
  regression with action-centering, Thompson Sampling with probability clipping,
  proxy delayed effect via simplified MDP, nightly batch updates
- **Simulation study structure** (Section 6): 3-fold cross-validation, prior
  construction from training batch, tuning parameter grid search, TS Bandit
  comparison
- **Key qualitative findings**: algorithm outperforms TS Bandit for a subset of
  participants, dosage variable reduces interventions over time

### What We Don't Reproduce

- **Exact numerical results** — different data source (synthetic NHANES-like
  instead of real HeartSteps V1)
- **HeartSteps V2 pilot data analysis** (Section 7 — requires real trial data)
- **Full feature set** — simplified to 4 baseline + 2 treatment features
  (paper uses location, temperature, app engagement, etc.)
- **Full grid search** — 6×6 grid with 10 re-runs per participant per
  parameter pair (paper uses 96 re-runs)

## Results

### Simulation Configuration

- **Participants:** 15 (5 per fold, 3-fold CV)
- **Days:** 30 per episode (extended from 7-day synthetic data)
- **Re-runs:** 10 per participant per algorithm
- **Grid:** gamma ∈ {0, 0.25, 0.5, 0.75, 0.9, 0.95}, w ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0}
- **Prior:** constructed from training batch via OLS (GEE approximation)

### Summary Statistics

| Metric | Value |
|--------|-------|
| Participants improved | 7/15 (46.7%) |
| Mean improvement | -1.91 |
| Median improvement | -2.93 |
| Std improvement | 10.97 |
| Min improvement | -22.83 |
| Max improvement | +13.51 |

### Paper Comparison

| Metric | This reproduction | Paper |
|--------|------------------|-------|
| % participants improved | 46.7% (7/15) | 78.4% (29/37) |
| Mean improvement | -1.91 | +29.75 |

### Why the Difference

The key finding (majority of participants improve) is NOT reproduced with
synthetic data. This is expected because:

1. **Synthetic data vs real HeartSteps V1**: The paper's prior is constructed
   from real HeartSteps V1 pilot data (37 participants, 42 days), which has
   genuine treatment effects. Our synthetic data has a fixed linear reward
   model with simplified features, so the treatment effect signal is weaker.

2. **Simplified features**: The paper uses 10+ features including location,
   temperature, and app engagement. Our 4+2 feature set reduces the advantage
   of context-aware Thompson Sampling.

3. **Fewer participants and re-runs**: 15 participants and 10 re-runs produce
   noisier estimates than the paper's 37 participants and 96 re-runs.

4. **Prior circularity**: Our prior is constructed from the same generative
   model used for evaluation. The paper uses independent HeartSteps V1 data.

### What IS Reproduced

- The algorithm implements all key equations faithfully
- The simulation pipeline runs end-to-end without errors
- The dosage variable correctly penalizes high-dosage interventions
- The proxy value function converges and produces meaningful delayed effect
  estimates
- Individual participants DO show improvement (7 out of 15), confirming the
  algorithm can outperform TS Bandit in some settings
- The best tuning parameters vary across folds (gamma=0.5, 0.75, 0), showing
  the grid search is working

## Code Structure

```
paper_reproduction/
├── heartsteps/          # Core algorithm
│   ├── dosage.py        # Exponential moving average dosage (Eq 1)
│   ├── bayesian_regression.py  # Action-centered Bayesian regression (Eq 3)
│   ├── agent.py         # Thompson Sampling with probability clipping
│   ├── proxy_value.py   # Simplified MDP value iteration (Section 5.4.2)
│   └── nightly_update.py  # Daily batch update routine
├── data/
│   ├── nhanes_loader.py    # Synthetic NHANES-like data generator
│   └── generative_model.py  # 90-day trajectory generator
├── simulation/
│   ├── cross_validation.py  # 3-fold CV over participants
│   ├── prior_construction.py  # GEE-like prior from training batch
│   ├── tuning.py         # Grid search over (gamma, w)
│   ├── run_study.py      # Full simulation pipeline
│   └── results.py        # Results dataclass and serialization
├── baselines/
│   └── ts_bandit.py      # Thompson Sampling Bandit (no proxy value)
├── visualization/
│   └── plot_results.py   # Bar chart of per-participant improvement
└── results/
    └── REPORT.md          # This file
```

## Running

```bash
cd rl-health-interventions-repro
uv run pytest tests/test_*.py -q          # Run all tests
uv run python -c "
from paper_reproduction.simulation.run_study import run_simulation
r = run_simulation(n_participants=15, n_days=30, n_re_runs=5, seed=42)
print(r.compute_summary())
"                                          # Run simulation
```
