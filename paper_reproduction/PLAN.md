# HeartSteps V2 Paper Reproduction Plan

> **Paper:** Liao, Greenewald, Klasnja, Murphy (2019). "Personalized HeartSteps: A Reinforcement Learning Algorithm for Optimizing Physical Activity." arXiv:1909.03539.

> **For Hermes:** Use subagent-driven-development to implement this plan task-by-task.

**Goal:** Faithfully reproduce the algorithm and simulation study from the HeartSteps V2 paper using publicly available step-count data, demonstrating that the algorithm works as described.

**What we reproduce:**
- The complete RL algorithm (Section 5): dosage variable, Bayesian linear regression with action-centering, Thompson Sampling with probability clipping, proxy delayed effect via simplified MDP, nightly batch updates
- The simulation study structure (Section 6): generative model, 3-fold cross-validation, tuning parameter grid search, comparison against TS Bandit
- Key qualitative findings: algorithm outperforms TS Bandit for majority of participants, dosage variable reduces interventions over time

**What we don't reproduce:**
- Exact numerical results (different data source — NHANES instead of HeartSteps V1)
- HeartSteps V2 pilot data analysis (Section 7 — requires real trial data)

**Data source:** NHANES minute-level step counts (CC0, 14.7K participants, 7 days each). We aggregate to 30-minute windows to match HeartSteps decision times.

**Architecture:** Self-contained `paper_reproduction/` package with no dependencies on the main framework's stubs. Clean implementation from scratch following the paper's equations exactly.

---

## Phase 1: Core Algorithm (Sections 5.2–5.4)

### Task 1.1: Dosage variable

**Objective:** Implement the exponential moving average dosage variable.

**Files:**
- Create: `paper_reproduction/heartsteps/dosage.py`
- Create: `tests/test_dosage.py`

**Equation:** X_{t+1} = lambda * X_t + 1{treatment or anti-sedentary suggestion}, lambda = 0.95

**Tests:**
- Dosage starts at 0
- Dosage increases by 1 when treatment delivered (before decay)
- Dosage decays by factor lambda when no treatment
- Anti-sedentary suggestions also increase dosage
- Dosage is always non-negative

### Task 1.2: Bayesian linear regression with action-centering

**Objective:** Implement the working model for treatment effect estimation.

**Files:**
- Create: `paper_reproduction/heartsteps/bayesian_regression.py`
- Create: `tests/test_bayesian_regression.py`

**Equation (3):** R_t = g(S_t)^T alpha_0 + pi_t * f(S_t)^T alpha_1 + (A_t - pi_t) * f(S_t)^T beta + N(0, sigma^2)

**Key detail:** Action-centering via (A_t - pi_t) makes treatment effect estimates robust to baseline model misspecification.

**Tests:**
- Posterior update produces correct mu and Sigma (verify against hand calculation)
- Action-centering: treatment effect estimate is unbiased even when baseline model is wrong
- Prior specification: significant features get population estimates, non-significant get zero
- Posterior variance decreases with more data

### Task 1.3: Thompson Sampling agent with probability clipping

**Objective:** Implement action selection using posterior samples.

**Files:**
- Create: `paper_reproduction/heartsteps/agent.py`
- Create: `tests/test_agent.py`

**Key details:**
- Sample beta from posterior N(mu_d, Sigma_d)
- Compute action value: Q(s, a) = r(s, a) + gamma * H(x, a) for a in {0, 1}
- Select action with probability proportional to exp(Q / temperature) or via TS
- Clip probabilities to [epsilon_1, epsilon_0] = [0.1, 0.2] range
- Available indicator I_t: if unavailable, force A_t = 0

**Tests:**
- Action selection returns 0 or 1
- Probabilities are clipped to [0.1, 0.2] range (in expectation over many samples)
- Unavailable state forces action 0
- Multiple calls with same state produce varied actions (stochastic)

### Task 1.4: Proxy delayed effect via simplified MDP

**Objective:** Implement the proxy value function for delayed effects.

**Files:**
- Create: `paper_reproduction/heartsteps/proxy_value.py`
- Create: `tests/test_proxy_value.py`

**Key details (Section 5.4.2):**
- Simplified MDP: context Z is i.i.d., availability I is i.i.d., dosage X transitions via tau(x'|x, a)
- tau(x'|x, 1) = 1{x' = lambda*x + 1} (treatment)
- tau(x'|x, 0) = p_sed * 1{x' = lambda*x + 1} + (1-p_sed) * 1{x' = lambda*x} (no treatment, p_sed=0.2)
- Solve Bellman equation for V*(x, i) on dosage grid
- Compute H*(x, a) = sum_{x',i'} tau(x'|x,a) * p(i'|avail) * V*(x', i')
- Weighted update: H_{d+1} = (1-w)*H_1 + w*H*
- Proxy delayed effect: eta(x) = gamma*H(x, 0) - gamma*H(x, 1)

**Tests:**
- H*(x, 1) < H*(x, 0) at high dosage (treatment reduces future value when over-dosed)
- eta(x) is negative at high dosage (signals to reduce interventions)
- eta(x) is near zero at low dosage (no penalty for fresh users)
- H_1 from HeartSteps V1 data is a reasonable initialization
- Bellman equation converges on a dosage grid

### Task 1.5: Nightly batch update

**Objective:** Wire together the nightly update routine.

**Files:**
- Create: `paper_reproduction/heartsteps/nightly_update.py`
- Create: `tests/test_nightly_update.py`

**Key details (Section 5.4):**
- Takes day's data: {S_{l,k}, A_{l,k}, R_{l,k}} for l=1..5 (5 decision times)
- Updates posterior (mu_{d+1}, Sigma_{d+1}) using equations (5) and (6)
- Updates proxy value H_{d+1} using grid search over dosage
- Returns updated parameters for next day

**Tests:**
- Update produces correct posterior given known data
- Proxy value update shifts H toward H* with correct weight
- Multiple days of updates show posterior variance decreasing

---

## Phase 2: Generative Model (Section 6.1)

### Task 2.1: NHANES data ingestion

**Objective:** Load and preprocess NHANES step counts into 30-minute windows.

**Files:**
- Create: `paper_reproduction/data/nhanes_loader.py`
- Create: `tests/test_nhanes_loader.py`

**Key details:**
- NHANES minute-level step counts (PhysioNet, CC0)
- Aggregate to 30-minute windows: sum of minute-level steps per window
- Extract context features: time of day (5 slots), day of week, prior 30-min steps, daily total
- Create participant-level datasets: each participant has ~7 days x 10 windows/day = 70 observations
- Select subset for simulation (e.g., 100 participants with sufficient data)

**Tests:**
- Loaded data has correct shape (n_participants, n_days, n_windows)
- Step counts are non-negative integers
- Context features are extractable
- No NaN values in processed data

### Task 2.2: Generative model construction

**Objective:** Build a generative model from NHANES data that simulates HeartSteps-like trajectories.

**Files:**
- Create: `paper_reproduction/data/generative_model.py`
- Create: `tests/test_generative_model.py`

**Key details (Section 6.1, "Generative model"):**
- For each participant i, construct a 90-day sequence of (Z_t, I_t, epsilon_t)
- Extend 7-day NHANES data to 90 days by concatenating randomly selected days
- Fit linear regression: R_t = g(S_t)^T alpha + A_t * f(S_t)^T beta + epsilon_t
- Treatment effect beta is the key parameter — set from GEE-like analysis of our data
- Availability probability p_avail estimated from data (assume ~0.85 based on HeartSteps V1)
- Anti-sedentary suggestion probability p_sed = 0.2

**Tests:**
- Generated trajectories have realistic step count distributions (mean, variance match NHANES)
- Treatment effect is non-zero and in expected range
- 90-day sequences are constructable from 7-day data
- Reward generation follows the specified linear model

---

## Phase 3: Simulation Study (Section 6)

### Task 3.1: Cross-validation infrastructure

**Objective:** Implement 3-fold cross-validation over participants.

**Files:**
- Create: `paper_reproduction/simulation/cross_validation.py`
- Create: `tests/test_cross_validation.py`

**Key details:**
- Partition participants into 3 folds
- For each iteration: 2 folds = training batch, 1 fold = testing batch
- Training batch used for: prior construction, noise variance estimation, tuning parameter selection
- Testing batch used for: algorithm evaluation

**Tests:**
- Each participant appears in exactly one test fold
- Training and test folds are disjoint
- 3 iterations cover all participants

### Task 3.2: Prior construction from training batch

**Objective:** Construct informative priors from training data using GEE-like procedure.

**Files:**
- Create: `paper_reproduction/simulation/prior_construction.py`
- Create: `tests/test_prior_construction.py`

**Key details (Section 5.5):**
- Fit population GEE (ordinary linear regression as approximation) using all training participants
- For significant features (p < 0.05): prior mean = point estimate, prior std = cross-participant std
- For non-significant features: prior mean = 0, prior std = half of cross-participant std
- For new features (e.g., app engagement): prior mean = 0, prior std = average of other stds
- Diagonal prior covariance matrices

**Tests:**
- Significant features get non-zero prior means
- Non-significant features get zero prior means with shrunk variance
- Prior matrices are diagonal
- Prior is reasonable (not too tight, not too loose)

### Task 3.3: Tuning parameter grid search

**Objective:** Select gamma and w via simulation-based grid search.

**Files:**
- Create: `paper_reproduction/simulation/tuning.py`
- Create: `tests/test_tuning.py`

**Key details (Section 6.1):**
- Grid: gamma in {0, 0.25, 0.5, 0.75, 0.9, 0.95}, w in {0, 0.1, 0.25, 0.5, 0.75, 1}
- For each (gamma, w) pair: run algorithm 96 times per training participant
- Select (gamma, w) that maximizes average total reward
- Use training batch participants' generative models

**Tests:**
- Grid search completes without errors
- Selected parameters are from the grid
- Average reward is computed correctly
- 96 re-runs produce stable estimates (low variance)

### Task 3.4: TS Bandit baseline

**Objective:** Implement the Thompson Sampling Bandit comparator.

**Files:**
- Create: `paper_reproduction/baselines/ts_bandit.py`
- Create: `tests/test_ts_bandit.py`

**Key details (Section 6):**
- Same as proposed algorithm but: maximizes immediate reward only (no proxy value)
- Same feature vectors, priors, noise variance, probability clipping
- Action selection: sample theta from posterior, select action maximizing r(s, a; theta)
- No action-centering in the reward model (uses r(s, a; theta) = g(s)^T alpha + a * f(s)^T beta)

**Tests:**
- TS Bandit selects actions based on immediate reward only
- Same prior construction as proposed algorithm
- Probability clipping applied
- Produces valid action sequences

---

## Phase 4: Run Simulation and Report Results

### Task 4.1: Full simulation runner

**Objective:** Wire together all components and run the complete simulation study.

**Files:**
- Create: `paper_reproduction/simulation/run_study.py`
- Create: `paper_reproduction/simulation/results.py`

**Key details:**
- For each CV iteration:
  1. Construct prior from training batch
  2. Estimate noise variance from training batch
  3. Grid search for (gamma, w) using training batch
  4. Build generative models for test batch participants
  5. Run proposed algorithm 96 times per test participant
  6. Run TS Bandit 96 times per test participant
  7. Compute average total reward for each participant under each algorithm
- Output: improvement per participant, average improvement, summary statistics

### Task 4.2: Results visualization and reporting

**Objective:** Generate Figure 2 equivalent and summary table.

**Files:**
- Create: `paper_reproduction/visualization/plot_results.py`
- Create: `paper_reproduction/results/REPORT.md`

**Key details:**
- Bar chart: improvement per participant (sorted), matching Figure 2
- Summary table: average improvement, number of participants improved
- Dose-response curves: intervention frequency over time for proposed vs TS Bandit
- Posterior mean traces for sample participants

---

## Phase 5: Verification

### Task 5.1: Subagent verification

**Objective:** Independent verification that results are reproducible.

**Process:**
- Spin up a fresh subagent with no prior context
- Give it: the paper PDF, the code, and instructions to run the simulation
- Subagent runs the full pipeline and checks:
  1. Algorithm implements paper's equations correctly
  2. Simulation study runs end-to-end without errors
  3. Results show algorithm outperforms TS Bandit for majority of participants
  4. Dosage variable reduces intervention frequency over time
  5. Code is well-tested and documented
- Subagent produces a verification report

### Task 5.2: Code quality audit

**Objective:** Final code quality pass.

**Process:**
- Run: `uv run ruff check paper_reproduction/`
- Run: `uv run ruff format paper_reproduction/`
- Run: `uv run pytest tests/ -v`
- Check: all tests pass, no lint errors, code is well-documented

---

## File Structure

```
paper_reproduction/
├── PLAN.md                          # This file
├── README.md                        # How to run
├── heartsteps/
│   ├── __init__.py
│   ├── dosage.py                    # Dosage variable (Task 1.1)
│   ├── bayesian_regression.py       # Bayesian regression with action-centering (Task 1.2)
│   ├── agent.py                     # TS agent with probability clipping (Task 1.3)
│   ├── proxy_value.py               # Proxy delayed effect (Task 1.4)
│   ├── nightly_update.py            # Nightly batch update (Task 1.5)
│   └── algorithm.py                 # Full algorithm wiring
├── data/
│   ├── __init__.py
│   ├── nhanes_loader.py             # NHANES data ingestion (Task 2.1)
│   └── generative_model.py          # Generative model from data (Task 2.2)
├── simulation/
│   ├── __init__.py
│   ├── cross_validation.py          # 3-fold CV (Task 3.1)
│   ├── prior_construction.py        # Prior from training batch (Task 3.2)
│   ├── tuning.py                    # Grid search for (gamma, w) (Task 3.3)
│   ├── run_study.py                 # Full simulation runner (Task 4.1)
│   └── results.py                   # Results computation
├── baselines/
│   ├── __init__.py
│   └── ts_bandit.py                 # TS Bandit baseline (Task 3.4)
├── visualization/
│   ├── __init__.py
│   └── plot_results.py              # Figure 2 equivalent (Task 4.2)
├── results/
│   └── REPORT.md                    # Final report
└── tests/
    ├── test_dosage.py
    ├── test_bayesian_regression.py
    ├── test_agent.py
    ├── test_proxy_value.py
    ├── test_nightly_update.py
    ├── test_nhanes_loader.py
    ├── test_generative_model.py
    ├── test_cross_validation.py
    ├── test_prior_construction.py
    ├── test_tuning.py
    └── test_ts_bandit.py
```

---

## Data Access

**NHANES minute-level step counts:**
- Source: PhysioNet (https://physionet.org/content/minute-level-step-count-nhanes/)
- License: CC0 (public domain)
- Format: CSV
- Size: 14,693 participants x 7 days x 1,440 minutes/day
- Access: Download directly, no application needed

**Download script:** `paper_reproduction/data/download_nhanes.py` (to be created)

We aggregate minute-level to 30-minute windows (10 windows/day matching HeartSteps 5 decision times x 2 halves).

---

## Success Criteria

1. **Algorithm fidelity:** All equations from Sections 5.2–5.4 implemented and tested
2. **Simulation runs:** Full 3-fold CV completes without errors
3. **Qualitative results:** Proposed algorithm outperforms TS Bandit for >50% of participants
4. **Dosage effect:** Intervention frequency decreases over time for responsive participants
5. **Code quality:** All tests pass, lint clean, well-documented
6. **Reproducibility:** Same seed produces identical results across runs
