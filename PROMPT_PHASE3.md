# Tasks 3.1-3.4: Simulation Study Infrastructure

You are building the simulation study for reproducing the HeartSteps V2 paper.

The worktree is at /Users/williamdennis/rl-health-interventions-repro
The package is at paper_reproduction/
Tests are at tests/

## Task 3.1: Cross-Validation Infrastructure

Create: paper_reproduction/simulation/cross_validation.py
Create: tests/test_cross_validation.py

Implement 3-fold cross-validation over participants (Section 6).

Key details:
- Partition participants into 3 roughly equal folds
- For each iteration i (i=1,2,3):
  - Training batch: folds {j, k} where j,k != i
  - Testing batch: fold {i}
- Training batch used for: prior construction, noise variance estimation, tuning parameter selection
- Testing batch used for: algorithm evaluation
- Each participant appears in exactly one test fold across all 3 iterations

Tests:
- Each participant appears in exactly one test fold
- Training and test folds are disjoint
- 3 iterations cover all participants
- Fold sizes are roughly equal

## Task 3.2: Prior Construction from Training Batch

Create: paper_reproduction/simulation/prior_construction.py
Create: tests/test_prior_construction.py

Construct informative priors from training data (Section 5.5).

Procedure:
1. Fit population linear regression (as GEE approximation) using all training participants
2. For each feature, assess significance (p < 0.05)
3. For significant features:
   - Prior mean = population regression point estimate
   - Prior std = standard deviation of person-specific estimates across participants
4. For non-significant features:
   - Prior mean = 0
   - Prior std = half of the cross-participant std
5. For new features (e.g., app engagement):
   - Prior mean = 0
   - Prior std = average of other features' prior stds
6. Construct diagonal prior covariance matrices

The prior is used to initialise the BayesianRewardModel.

Tests:
- Significant features get non-zero prior means
- Non-significant features get zero prior means with shrunk variance
- Prior matrices are diagonal
- Prior dimensions match the model (g_dim + 2*f_dim)

## Task 3.3: Tuning Parameter Grid Search

Create: paper_reproduction/simulation/tuning.py
Create: tests/test_tuning.py

Select gamma and w via simulation-based grid search (Section 6.1).

Grid:
- gamma (discount rate): {0, 0.25, 0.5, 0.75, 0.9, 0.95}
- w (update weight): {0, 0.1, 0.25, 0.5, 0.75, 1}

For each (gamma, w) pair:
1. Run the proposed algorithm 96 times per training participant
2. Compute average total reward across re-runs and participants
3. Select (gamma, w) that maximizes average total reward

Note: For initial implementation, use a small number of re-runs (e.g., 10) and participants (e.g., 10) for speed. The full 96 re-runs can be used for final results.

Tests:
- Grid search completes without errors
- Selected parameters are from the grid
- Average reward is computed correctly
- Results are deterministic with same seed

## Task 3.4: TS Bandit Baseline

Create: paper_reproduction/baselines/ts_bandit.py
Create: tests/test_ts_bandit.py

Implement the Thompson Sampling Bandit comparator (Section 6).

Key difference from proposed algorithm:
- Maximizes IMMEDIATE reward only (no proxy value)
- Same feature vectors, priors, noise variance, probability clipping
- Action selection: sample theta from posterior, select action maximizing r(s, a; theta)
- No action-centering in the reward model (uses r(s, a; theta) = g(s)^T alpha + a * f(s)^T beta)

The TS Bandit is the baseline for comparison.

Tests:
- TS Bandit selects actions based on immediate reward only
- Same prior construction as proposed algorithm
- Probability clipping applied
- Produces valid action sequences
- Runs without errors on synthetic data

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
- Use print() for output
- Skip writing tests
