# HeartSteps V2 Reproduction — Verification Report

**Date:** 2026-06-12  
**Verifier:** Independent subagent (no prior project knowledge)  
**Paper:** Liao, Greenewald, Klasnja, Murphy (2019), arXiv:1909.03539  
**Codebase:** `/Users/williamdennis/rl-health-interventions-repro/paper_reproduction/`

---

## A. Algorithm Fidelity

### A.1 Dosage Variable (Section 5.2, Equation 1)

**Paper:** X_{t+1} = lambda * X_t + 1{treatment or anti-sedentary suggestion}, lambda = 0.95  
**Code:** `heartsteps/dosage.py` — `DosageTracker.update()` line 74: `self._value = self.decay * self._value + event_int`  
**Verdict:** PASS. Equation implemented correctly. Supports both `treatment_delivered` and `anti_sedentary_delivered` flags.

### A.2 Bayesian Linear Regression with Action-Centering (Section 5.4.1, Equation 3)

**Paper:** R_t = g(S_t)^T alpha_0 + pi_t * f(S_t)^T alpha_1 + (A_t - pi_t) * f(S_t)^T beta + N(0, sigma^2)  
**Code:** `heartsteps/bayesian_regression.py` — `construct_features()` line 133: `phi = np.concatenate([g, pi * f, centering * f])`  
**Verdict:** PASS. Action-centering via (A_t - pi_t) correctly implemented. Posterior update uses equations (5)-(6).

### A.3 Thompson Sampling with Probability Clipping (Section 5.3)

**Paper:** Sample beta ~ N(mu_d, Sigma_d), compute Q(s, a), softmax, clip to [0.1, 0.2]  
**Code:** `heartsteps/agent.py` — `select_action()` lines 120-121: `prob = float(np.clip(prob_raw, self.epsilon_1, self.epsilon_0))`  
**Verdict:** PASS. Clipping defaults [epsilon_1=0.1, epsilon_0=0.2] match paper.

### A.4 Proxy Delayed Effect via Simplified MDP (Section 5.4.2)

**Paper:** Simplified MDP with dosage transitions tau(x'|x, a), value iteration, H(x, a), eta(x) = gamma*H(x,0) - gamma*H(x,1)  
**Code:** `heartsteps/proxy_value.py` — `solve()` performs value iteration on dosage grid. `eta()` computes delayed effect.  
**Verdict:** PASS. Transition dynamics and Bellman equation match paper.

### A.5 Nightly Batch Updates (Section 5.4)

**Paper:** After each day, update posterior + re-solve proxy value function with weighted blend  
**Code:** `heartsteps/nightly_update.py` — `update()` processes day's observations, calls `proxy_value.solve()` then `proxy_value.update()`  
**Verdict:** PASS. Structure matches paper.

---

## B. Simulation Study

### B.1 Pipeline Execution

**Command:** `cd /Users/williamdennis/rl-health-interventions-repro && uv run python -c "from paper_reproduction.simulation.run_study import run_simulation; r = run_simulation(n_participants=9, n_days=15, n_re_runs=2, seed=42); ..."`

**Verdict:** PASS. The pipeline runs end-to-end without errors. 3-fold CV completes, prior construction, grid search, and evaluation all execute successfully.

### B.2 Test Suite

**Command:** `uv run pytest tests/test_*.py -v` (all 11 test files, 104 tests + 8 tuning tests = 112 total)

**Verdict:** PASS. All 112 tests pass (104 main + 8 tuning).

### B.3 Test Coverage Observations

- Tests verify posterior updates against hand calculations (good)
- Tests verify action-centering robustness to baseline misspecification (good)
- Tests verify probability clipping (good)
- Tests verify Bellman convergence (good)
- Tests verify prior construction significance detection (good)
- Tests verify CV fold disjointness (good)
- Tests verify determinism across seeds (good)

---

## C. Key Finding: Does Proposed Algorithm Outperform TS Bandit?

### C.1 Simulation Results (n_participants=9, n_days=15, n_re_runs=2)

```
Improvements: [(2, -1.39), (5, 5.58), (1, -25.69), (6, -3.67), (7, 18.2), (0, 5.8), (8, 5.53), (4, -20.74), (3, -11.17)]
Summary: n_improved=4/9 (44.4%), mean_improvement=-3.06
```

### C.2 Paper's Reported Result

"for the majority of participants (29 out of 37), the total rewards is higher comparing with TS Bandit algorithm. The average improvement of the total rewards over TS Bandit is 29.753."

### C.3 Verdict

**FAIL.** The reproduction does NOT match the paper's qualitative finding. Only 44.4% of participants improved (vs 78.4% in the paper), and the mean improvement is negative (-3.06 vs +29.753 in the paper). The claim in `results/REPORT.md` that "the proposed algorithm outperforms the TS Bandit for more than 50% of participants" is FALSE based on the actual simulation output.

**NOTE:** The low number of re-runs (2 vs 96 in the paper) and participants (9 vs 37) inflates noise, but even accounting for noise, the direction does not match.

---

## D. Dosage Effect: Does Dosage Reduce Intervention Frequency?

This was not explicitly tested in the simulation run I performed, but based on code analysis:

- The proxy value function eta(x) = gamma * (H(x,0) - H(x,1)) is **non-negative at all dosage levels** (confirmed by test `test_eta_positive_all_dosages`)
- This means at high dosage, H(x,0) >= H(x,1), so the proxy penalizes sending more than not sending
- At low dosage, the effect is also non-negative, meaning the agent is always slightly penalized for sending
- The penalty is added to Q-values: `Q1 = g@alpha_0 + pi*(f@alpha_1) + (1-pi)*(f@beta) + gamma*H(x,1)` vs `Q0 = g@alpha_0 + 0 + (0-pi)*(f@beta) + gamma*H(x,0)`
- Since H(x,0) >= H(x,1), the proxy term `gamma*H(x,a)` always favors not sending, with larger effect at higher dosage
- Combined with probability clipping [0.1, 0.2], the max intervention probability is capped at 0.2 regardless

**Verdict:** PASS (by design). The dosage variable mechanically reduces intervention frequency over time due to the proxy value function penalizing action 1 when dosage is high. But this is a feature of the MDP design, not an empirical result that was checked.

---

## E. Code Quality

### E.1 Comprehensiveness

- **11 test files** covering all modules
- **112 passing tests** — good coverage of edge cases
- Tests include: shape checks, value checks, determinism, convergence, robustness to model misspecification

### E.2 Documentation

- All source files have detailed docstrings with paper references (arXiv number, section numbers)
- PLAN.md is comprehensive with task breakdown, equations, and test specifications
- results/REPORT.md provides methodology description and interpretation

### E.3 Code Organization

- Clean package structure: `heartsteps/`, `data/`, `simulation/`, `baselines/`, `visualization/`
- Each component is in its own file with clear responsibilities
- Type hints throughout
- No lint errors visible

**Verdict:** PASS (good code quality).

---

## F. Issues Found

### F.1 [BUG] Anti-Sedentary Suggestions Not Included in Dosage During Simulation

**Severity:** HIGH  
**Location:** `heartsteps/agent.py` line 146-149, `simulation/run_study.py`  
**Description:** The paper's generative model (Section 6.1, steps 1-2) specifies that dosage X_t depends on event E_t = {A_{t-1}=1} ∪ {B_t=1}, where B_t is an anti-sedentary suggestion indicator (probability 0.2). However, in the simulation, the agent's `step()` method only passes `treatment_delivered` to the dosage tracker, never `anti_sedentary_delivered`. The dosage is therefore systematically lower than it should be, which weakens the proxy value's ability to penalize over-intervention.

```python
# agent.py line 146-149:
treatment_delivered = available and (action == 1)
self.dosage_tracker.update(
    treatment_delivered=treatment_delivered,
    # MISSING: anti_sedentary_delivered is never passed
)
```

The `DosageTracker.update()` method accepts `anti_sedentary_delivered` but it's never used with True in any simulation episode.

### F.2 [FAIL] Key Qualitative Result Not Reproduced

**Severity:** HIGH  
**Description:** The paper reports 29/37 (78.4%) of participants show improvement with mean +29.753. The actual simulation shows 4/9 (44.4%) with mean -3.06. While acknowledging different data and fewer participants, the directional finding (majority improved) is not reproduced. The `results/REPORT.md` incorrectly states the finding IS reproduced ("majority of participants show improvement").

### F.3 [DEVIATION] Reduced Grid Search and Re-Runs

**Severity:** MEDIUM  
**Location:** `simulation/run_study.py` default parameters  
**Description:** 
- Paper uses gamma grid {0, 0.25, 0.5, 0.75, 0.9, 0.95} and w grid {0, 0.1, 0.25, 0.5, 0.75, 1} with 96 re-runs
- Code defaults: gamma=(0, 0.5, 0.9), w=(0, 0.5, 1.0), n_re_runs=3
- The smaller grid and fewer re-runs produce noisier results and may not find the optimal (gamma, w)

### F.4 [DEVIATION] Synthetic NHANES Data Instead of HeartSteps V1

**Severity:** MEDIUM  
**Description:** The paper uses real HeartSteps V1 data (37 participants, 42 days). The reproduction uses synthetic NHANES-like data with simplified activity patterns. This is acknowledged in PLAN.md but is a major methodological difference that explains the lack of result replication.

### F.5 [DEVIATION] Simplified Feature Vectors

**Severity:** MEDIUM  
**Description:** The paper uses features including location, temperature, app engagement, step variation, prior 30-min steps, and yesterday's steps. The code uses a simplified set of 4 baseline features and 2 treatment features. This reduces the advantage of action-centering and context-aware TS.

### F.6 [DEVIATION] OLS Instead of GEE for Prior Construction

**Severity:** LOW  
**Description:** The paper uses Generalized Estimating Equations (GEE) for prior construction. The code uses OLS as an approximation. This is explicitly noted in `prior_construction.py` line 6: "Fit population OLS (as GEE approximation)". OLS does not account for within-participant correlation.

### F.7 [MISSING] `algorithm.py` Not Created

**Severity:** LOW  
**Location:** `heartsteps/`  
**Description:** PLAN.md lists `algorithm.py` for "Full algorithm wiring" but the file does not exist. The algorithm wiring is distributed across `agent.py`, `nightly_update.py`, and `run_study.py`.

### F.8 [MISSING] README.md Not Created

**Severity:** LOW  
**Description:** PLAN.md and pyproject.toml reference a `README.md` but no such file exists in `paper_reproduction/`.

### F.9 [ISSUE] Prior Construction Uses Same Generative Model as Evaluation

**Severity:** MEDIUM  
**Description:** The prior is constructed by generating trajectories from a generative model that was built from the SAME data used for evaluation. In the paper, the prior is built from independent HeartSteps V1 data. This circularity means the prior is optimally matched to the generative model, which could bias results.

---

## G. Summary

| Criteria | Status | Notes |
|---|---|---|
| A. Algorithm Fidelity | PASS | All key equations implemented |
| B. Simulation Runs | PASS | End-to-end without errors |
| C. Key Finding (outperforms TS Bandit) | **FAIL** | 44.4% vs paper's 78.4% |
| D. Dosage Effect | PASS | Mechanism correct via proxy value |
| E. Code Quality | PASS | Well-tested, well-documented |
| F. Issues Found | **ISSUES DETECTED** | See Section F |

**Bottom Line:** The code faithfully implements the paper's algorithm and the simulation pipeline runs without errors. However, the **key qualitative finding is NOT reproduced** — the proposed algorithm does NOT outperform the TS Bandit for the majority of participants in this reproduction. The primary causes appear to be: (1) the anti-sedentary dosage bug (F.1), (2) synthetic data vs real HeartSteps V1 data (F.4), (3) simplified features (F.5), and (4) reduced grid search / re-runs (F.3). The `results/REPORT.md` makes a false claim about the results.
