# Issue 6: Write simulator design section for Phase 2 (documentation)

**Labels:** `documentation`
**Assignees:** @wd7512

## Context

Mengyan asked for this **four times** during the design doc review:

1. *"I'd suggest having a separate section for the simulator, and write in detail the design of a simulator: data to use, what the input of the simulator, what's the simulator design (model, probability design, etc), what the simulator can do (and how we can test whether it's valid)"*
2. *"I think we'd need more details, also for phase 2, e.g. data to use, what the input of the simulator, what's the simulator design"*
3. *"It remains unclear how you propose to train the simulator"*
4. *"If we do not have the real environment, then we'd need a simulator to simulate user response/body data. I imagine it would be a model (e.g. neural network, or generative model) which takes the state, action, user context in, and outputs the next state, user response/data changes, reward, etc."*

The existing docs (`initial_design.tex`, `ROADMAP.md`) cover the MDP environment and rule-based transitions but do **not** have a dedicated section describing the learned transition model architecture, training data requirements, or validation strategy for Phase 2.

## Specification

Add a new section to `docs/design/initial_design.tex` (e.g., **§6: Simulator Design for Learned Transitions**) covering:

### 6.1 Simulator role and interface

- Clear definition: the simulator is a function `(s_t, a_t, θ_user) → (s_{t+1}, r_t)` that produces the next state and reward given the current state, action, and user parameters
- Distinction from the MDP environment wrapper (which calls the simulator and tracks episode state)
- Distinction from Phase 1's rule-based transition (hand-crafted matrix vs learned from data)

### 6.2 Architectural options (with feasibility assessment)

| Architecture | Complexity | Data needed | Strengths | Weaknesses |
|-------------|-----------|-------------|-----------|------------|
| Fixed transition matrix (Phase 1) | None | None (hand-curated) | Simple, interpretable | No personalisation, no learning |
| Probabilistic graphical model (Bayesian network) | Low | Small (N=50 users) | Interpretable, uncertainty-aware | Limited expressivity |
| MLP / RNN neural network | Medium | Medium (N=500+ users) | Flexible, can capture non-stationarity | Needs more data, less interpretable |
| Generative model (diffusion/VAE) | High | Large (N=5000+ users) | Multi-modal distributions, realistic | Complex training, overkill for Phase 2 |
| LLM-based (bootstrapped) | Low | None (uses LLM prior) | Quick, zero data | LLM biases, not grounded in real data |

Recommended approach for Phase 2: start with a Bayesian network or small MLP, then scale up based on data availability.

### 6.3 Training data requirements

For each Phase 2 dataset, assess:

| Dataset | Has `(s, a, s', r)` structure? | N users | N timesteps | Access | Suitable for training? |
|---------|-------------------------------|---------|-------------|--------|----------------------|
| TILES | ✅ (intervention delivery logs + Fitbit) | ~200 | 12 months | Open (OSF) | ✅ Best immediate option |
| HeartSteps V1 | ✅ (MRT with 5×/day decisions) | ~50 | 42 days | Weeks-months | ✅ Small but high-quality |
| HeartSteps V2 | ✅ (RL-in-the-loop TS) | ~97 | 90 days | Weeks-months | ✅ Larger, more actions |
| All of Us | ❌ (passive sensing, no interventions) | 59K | 14 years | 4-8 weeks | ❌ No action→response data |
| UK Biobank | ❌ (passive accelerometry) | 100K | 7 days | 4-8 weeks | ❌ No action→response data |
| NHANES | ❌ (passive step counts) | 14.7K | 7 days | Open (CC0) | ❌ No action→response data |

**Key finding:** Only TILES, HeartSteps V1, and HeartSteps V2 contain the intervention delivery logs needed to train a transition function. All of Us, UK Biobank, and NHANES can only provide population distributions for synthetic data parameterisation.

### 6.4 Training procedure

- Data preprocessing: align timesteps, handle missing data, encode categorical features
- Train/val/test split: by user (not by timestep) — avoid data leakage
- Loss function: cross-entropy for discrete states, MSE for continuous variables
- Evaluation metrics: predictive accuracy, KL divergence from held-out trajectories, calibration of uncertainty

### 6.5 Validation criteria

How do we test whether the simulator is valid?
1. **Distributional match:** simulated trajectories have similar step count distributions, autocorrelation, and day-of-week patterns as real data
2. **Response curve:** treatment effect estimates from simulated RCTs match published literature (e.g., HeartSteps effect sizes)
3. **Agent ranking preservation:** the ordering of agent performance is stable across simulator versions
4. **Sensitivity analysis:** varying transition parameters produces expected changes in outcomes

### 6.6 Data gaps and mitigation

- `goal_progress` and `burden` are state variables that can be computed deterministically from action and step history
- `body_measure` needs EHR linkage (All of Us has 46% linked)
- Previous action/response must be tracked as internal state, not from data

### Deliverables

1. New **§6 Simulator Design** section in `docs/design/initial_design.tex` with all subsections above
2. Updated `docs/design/initial_design.pdf` (rebuild with `pdflatex`)
3. Updated `docs/sources/data_sources.md` and `docs/sources/additional_data_sources.md` with a note linking to the new section

## Out of scope
- Implementation of any learned transition model (Phase 2, after data access)
- Real data downloads or access requests (tracked in Issue #48)

## Related
- `docs/design/initial_design.tex` (existing doc to extend)
- `docs/sources/data_sources.md` (existing dataset analysis)
- `docs/sources/additional_data_sources.md` (additional dataset analysis)
- Issue #48 (external data access blockers)
- Issue #5 (LLM bootstrapping — one architectural option for the simulator)
