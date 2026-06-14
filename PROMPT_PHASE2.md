# Tasks 2.1-2.2: NHANES Data Ingestion and Generative Model

You are building the data layer for reproducing the HeartSteps V2 paper simulation study.

The worktree is at /Users/williamdennis/rl-health-interventions-repro
The package is at paper_reproduction/
Tests are at tests/

## Task 2.1: NHANES Data Ingestion

Create: paper_reproduction/data/nhanes_loader.py
Create: tests/test_nhanes_loader.py

The NHANES minute-level step count dataset is available from PhysioNet:
- URL: https://physionet.org/content/minute-level-step-count-nhanes/
- License: CC0 (public domain)
- Format: CSV files with minute-level step counts for 14,693 participants over 7 days

For the simulation, we need:
1. Download a subset of NHANES data (or use synthetic data that mimics its structure)
2. Aggregate minute-level counts to 30-minute windows (10 windows per day)
3. Extract context features matching HeartSteps:
   - Time of day (encoded as 5 decision time slots, each ~2.5 hours)
   - Day of week
   - Prior 30-minute step count
   - Yesterday's total step count
4. Create participant-level datasets

Since we may not have the actual NHANES data available immediately, create:
a) A SyntheticNHANESGenerator that produces realistic step count data:
   - Step counts follow a distribution matching published NHANES statistics
   - Mean ~5000 steps/day for sedentary adults, std ~2000
   - Time-of-day patterns: morning peak, afternoon dip, evening activity
   - 7-day sequences with realistic day-to-day variation
b) A loader interface that can switch between synthetic and real data

The synthetic generator should produce data with these properties:
- n_participants participants, each with n_days days
- Each day has 10 thirty-minute windows
- Step counts are non-negative integers
- Realistic time-of-day patterns (higher in morning/evening)
- Day-to-day correlation (autoregressive)

Tests:
- Output has correct shape (n_participants, n_days, n_windows)
- Step counts are non-negative
- Time-of-day patterns are realistic (morning > night)
- No NaN values
- Deterministic with same seed

## Task 2.2: Generative Model Construction

Create: paper_reproduction/data/generative_model.py
Create: tests/test_generative_model.py

Build a generative model from the NHANES-like data that simulates HeartSteps-like trajectories (Section 6.1).

The generative model for participant i produces a 90-day sequence of:
- Context Z_t (time of day, day of week, prior steps, etc.)
- Availability I_t (binary, p_avail ≈ 0.85)
- Residual epsilon_t (from person-specific regression)
- Reward R_t given action A_t

Construction procedure:
1. From 7-day NHANES data, extend to 90 days by concatenating randomly selected days
2. Fit a linear regression model: R_t = g(S_t)^T * alpha + A_t * f(S_t)^T * beta + epsilon_t
   - g(s): baseline features (prior 30-min steps, daily total, time of day, temperature)
   - f(s): treatment effect features (dosage, location, step variation)
   - alpha, beta: regression coefficients
   - epsilon_t: residual noise
3. The treatment effect beta is key — set it to a reasonable value (e.g., 50-200 extra steps per suggestion)
4. Availability probability p_avail estimated from data (assume 0.85)
5. Anti-sedentary suggestion probability p_sed = 0.2

The generative model is used to:
- Create training data for prior construction
- Create test data for algorithm evaluation
- Run the simulation study

Tests:
- Generated trajectories have realistic step count distributions
- Treatment effect is non-zero and in expected range (50-200 steps)
- 90-day sequences are constructable from 7-day data
- Reward generation follows the linear model
- Different participants have different response patterns

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
- Modify existing heartsteps/ files
- Add new dependencies
- Use print() for output
- Skip writing tests
