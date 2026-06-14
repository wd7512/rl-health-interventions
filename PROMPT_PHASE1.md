# Tasks 1.3-1.5: Thompson Sampling Agent, Proxy Delayed Effect, Nightly Update

You are implementing parts of the HeartSteps V2 RL algorithm from:
Liao et al. (2019). "Personalized HeartSteps." arXiv:1909.03539

The worktree is at /Users/williamdennis/rl-health-interventions-repro
The package is at paper_reproduction/heartsteps/
Tests are at tests/

Already implemented (DO NOT modify):
- paper_reproduction/heartsteps/dosage.py — DosageTracker class
- paper_reproduction/heartsteps/bayesian_regression.py — BayesianRewardModel class
- tests/test_dosage.py — 11 tests passing
- tests/test_bayesian_regression.py — 10 tests passing

## Task 1.3: Thompson Sampling Agent with Probability Clipping

Create: paper_reproduction/heartsteps/agent.py
Create: tests/test_agent.py

The agent selects actions using Thompson Sampling with probability clipping.

Algorithm (Section 5.3, Equation 2):
1. If unavailable (I_t = 0): set A_t = 0
2. If available (I_t = 1):
   a. Sample beta_sample from posterior N(mu_d, Sigma_d) using model.sample_beta()
   b. Compute action values:
      - Q(s, 0) = g(s)^T * alpha_0_hat + 0 * f(s)^T * alpha_1_hat + (0 - pi) * f(s)^T * beta_sample + gamma * H(dosage, 0)
      - Q(s, 1) = g(s)^T * alpha_0_hat + pi * f(s)^T * alpha_1_hat + (1 - pi) * f(s)^T * beta_sample + gamma * H(dosage, 1)
      where H(dosage, a) is the proxy future value from Task 1.4
   c. Select action with probability:
      pi_t = clip(exp(Q(s,1)/tau) / (exp(Q(s,0)/tau) + exp(Q(s,1)/tau)), epsilon_1, epsilon_0)
      where tau is temperature, epsilon_0=0.2, epsilon_1=0.1
   d. Sample A_t ~ Bernoulli(pi_t)

Key details:
- epsilon_0 = 0.2 (max probability of sending suggestion)
- epsilon_1 = 0.1 (min probability — ensures exploration)
- tau = 1.0 (temperature, can be configurable)
- The agent needs access to the BayesianRewardModel and DosageTracker
- The agent needs the proxy value function (Task 1.4) for the delayed effect term

Tests:
- Action is 0 or 1
- Probabilities are clipped to [0.1, 0.2] range
- Unavailable state forces action 0
- Multiple calls with same state produce varied actions
- Agent updates posterior after each step

## Task 1.4: Proxy Delayed Effect via Simplified MDP

Create: paper_reproduction/heartsteps/proxy_value.py
Create: tests/test_proxy_value.py

This computes the proxy for delayed effects on future rewards (Section 5.4.2).

The simplified MDP:
- State: (dosage x, availability i)
- Context Z is i.i.d. (not modeled in transitions)
- Availability I is i.i.d. with probability p_avail
- Dosage transitions:
  - tau(x'|x, 1) = 1{x' = lambda*x + 1}  (treatment)
  - tau(x'|x, 0) = p_sed * 1{x' = lambda*x + 1} + (1-p_sed) * 1{x' = lambda*x}  (no treatment)
  - p_sed = 0.2 (anti-sedentary suggestion probability)

The Bellman equation for V*(x, i):
  V(x, i) = max_{a in A(i)} [r(x, a) + gamma * sum_{x',i'} tau(x'|x,a) * p(i'|avail) * V(x', i')]
where:
  - A(1) = {0, 1}, A(0) = {0}
  - r(x, a) = marginal reward (depends only on dosage and action)
  - gamma = discount rate (tuning parameter)

The proxy delayed effect:
  eta(x) = gamma * H(x, 0) - gamma * H(x, 1)
where H(x, a) = sum_{x',i'} tau(x'|x,a) * p(i'|avail) * V*(x', i')

Weighted update:
  H_{d+1} = (1-w) * H_1 + w * H*
where H_1 is the initial proxy from HeartSteps V1 data, w is a tuning parameter.

Implementation approach:
- Discretize dosage to a grid (e.g., 0, 0.5, 1.0, ..., 10.0)
- Solve Bellman equation via value iteration on the grid
- H_1 can be initialized from the data (or set to zeros initially)
- The update() method takes new data and recomputes H*

Tests:
- H*(x, 1) < H*(x, 0) at high dosage (treatment reduces future value when over-dosed)
- eta(x) is negative at high dosage (signals to reduce interventions)
- eta(x) is near zero at low dosage (no penalty for fresh users)
- Bellman equation converges on a dosage grid
- Weighted update blends H_1 and H* correctly

## Task 1.5: Nightly Batch Update

Create: paper_reproduction/heartsteps/nightly_update.py
Create: tests/test_nightly_update.py

Wires together the nightly update routine (Section 5.4).

The nightly update takes a full day's data and updates all model components:
- Input: {S_{l,k}, A_{l,k}, R_{l,k}} for l=1..5 (5 decision times on day k)
- Updates BayesianRewardModel posterior (equations 5-6)
- Updates proxy value H using the new data
- Returns updated parameters for next day

The update uses only available decision times (I_{l,k} = 1).

Tests:
- Update produces correct posterior given known data
- Proxy value update shifts H toward H* with correct weight
- Multiple days of updates show posterior variance decreasing
- Unavailable decision times are correctly skipped

## Code Quality Requirements
- Follow existing code style in dosage.py and bayesian_regression.py
- Use logging (stdlib) for all output, never print()
- Type hints on all public methods
- Docstrings on all public classes and methods
- Reference paper equations in comments where applicable
- Run: uv run ruff check paper_reproduction/ tests/
- Run: uv run ruff format paper_reproduction/ tests/
- Run: uv run pytest tests/ -v
- All tests must pass before finishing

## DO NOT:
- Modify existing files (dosage.py, bayesian_regression.py, test_dosage.py, test_bayesian_regression.py)
- Add new dependencies
- Use print() for output
- Skip writing tests
