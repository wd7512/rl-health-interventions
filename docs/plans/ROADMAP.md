---
title: "Roadmap — rl-health-interventions"
updated: "2026-06-17"
---

# Roadmap — rl-health-interventions

## Vision Statement

Build a configurable simulation framework where RL-driven health intervention experiments are specified entirely through YAML config files, enabling systematic cross-dataset and cross-policy comparisons without bespoke engineering. The framework ships features as supervisors need them; the longer-term goal is a working CLI that trains agents on synthetic data and produces reproducible results tables, with Nature Methods publication readiness downstream of that.

## Non-Goals (Current Phase)

Parked until the MVP (Issue #101) is stable and we have supervisor
feedback on it:

- Multi-timescale reward (immediate + delayed body measure)
- 4 user archetypes (goal-driven, social, resistant, stable)
- Burden accumulation / decay model
- Evaluation framework (bootstrap CIs, power analysis)
- Multi-feature synthetic data matched to population statistics
- Safety / ethics review (pending real data)

These belong in the backlog. They are not "blocked" — they are
intentionally not the current focus.

## Milestone Backlog

Rough guidance, not a release plan. Active work is tracked in
GitHub issues with the `phase-1` label. This table describes the
eventual shape of the framework, not the order it will be built in.

| ID | Name | Status | Description | Prerequisite IDs |
|----|------|--------|-------------|-----------------|
| M-01 | Config Schema & Validation | ✅ Completed | Pydantic schemas (MDPConfig, AgentConfig) with cross-reference validators, schema-reference mode stub, precomputed per-step reward | — |
| M-02 | StateView & Environment | ✅ Completed | StateView dataclass (activity, day, step_of_day) and Environment with step/reset using registry factories | M-01 |
| M-03 | Transition & Reward Models | ✅ Completed | RuleBasedTransition (config-driven matrix) and CompoundReward (precomputed per-step reward) | M-02 |
| M-04 | User Simulation Engine | 🔜 Planned | Implement UserProfile with 4 archetypes and RuleBasedResponse model | M-03 |
| M-05 | Thompson Sampling Agent | ✅ Completed | ThompsonSamplingAgent with Beta-Bernoulli posterior (configurable alpha/beta priors). Also includes epsilon-greedy, UCB, and random baselines | M-02 |
| M-06 | Experiment Runner & CLI | ✅ Completed | `run_experiment()` iterates agents from config; CLI accepts `--config`, `--agent`, `--output`, `--seed`; regression test with JSON fixture | M-01, M-05 |
| M-07 | Synthetic Data Pipeline | 🔜 Planned | Complete synthetic data generation with realistic wearable data distributions | M-01 |
| M-08 | Evaluation Framework | 🔜 Planned | Define baselines, metrics, and statistical analysis plan for agent comparison | M-06 |
| M-09 | Documentation & Examples | 🔜 Planned | Complete API docs, example configs, and contributor guide | M-06 |
| M-10 | Safety & Ethics Review | 🔜 Planned | Add safety constraints, privacy documentation, and ethics considerations | M-03, M-04 |

## Milestone Detail Cards

### M-01: Config Schema & Validation ✅
**Objective:** Pydantic config schema with cross-reference validators.

**Deliverables (completed):**
- `config/schemas.py` with MDPConfig, AgentConfig, TransitionModelConfig, TransitionProbabilities
- String-keyed states/actions (no enums), cross-reference validators, schema-reference mode stub
- `config/loader.py` for YAML → Pydantic loading
- Precomputed `per_step_reward` via Pydantic `@model_validator`
- Agent parameter validation (type-specific required fields)

**Out of scope:** DataConfig, ExperimentConfig, 3-layer validation — deferred to Phase 2.

**Dependencies:** None

---

### M-02: StateView & Environment ✅
**Objective:** StateView dataclass and Environment with step/reset.

**Deliverables (completed):**
- `state.py` with frozen StateView dataclass (activity, day, step_of_day, steps_per_day, global_step)
- `environment.py` with Environment.step(action) and reset() using registry factories (`make_transition`, `make_reward`)
- String-keyed activity tracking (no ActivityLevel enum)
- Fixed-length episode termination (steps_per_day × episode_days)

**Out of scope:** Multi-timescale reward, UserProfile-based transitions, `from_dataset()` — deferred to Phase 2.

**Dependencies:** M-01

---

### M-03: Transition & Reward Models ✅
**Objective:** RuleBasedTransition (config-driven matrix) and CompoundReward (precomputed per-step reward).

**Deliverables (completed):**
- `transitions/rule_based.py` with RuleBasedTransition using the config's transition probability matrix (string-keyed)
- `rewards/compound.py` with CompoundReward using precomputed `per_step_reward` array indexed by (state_name, step_idx)
- Registry factories (`make_transition`, `make_reward`) wired into Environment
- Validation: transition probabilities must sum to 1.0 (tolerance 1e-6);
 `reward_multiplier_by_step` length must match steps_per_day

**Out of scope:** UserProfile-based transitions, burden accumulation/decay, multi-timescale reward, configurable reward weights — deferred to Phase 2.

**Dependencies:** M-02

---

### M-04: User Simulation Engine
**Objective:** Implement UserProfile with 4 behavioural archetypes and RuleBasedResponse model producing distinct response patterns.

**Deliverables:**
- `simulation/user_profile.py` with UserProfile Pydantic schema
- `simulation/rule_based.py` with RuleBasedResponse implementing archetype-specific responses
- 4 archetypes: goal-driven, social responder, resistant, stable maintainer
- Parameter ranges for each archetype (response magnitude, burden rate, baseline activity)
- Response model tests (each archetype produces expected response direction)

**Definition of Done:** 4 archetypes produce distinct, plausible response patterns; goal-driven responds to reminders, social responds to motivation, resistant shows flat response, stable shows low marginal gain; burden threshold triggers response decay; all parameters configurable.

**Dependencies:** M-03 (Transition model uses response model output)

**Risks:**
- Risk: Archetype parameters not grounded in clinical literature. Mitigation: Cite StepCountJITAI and HeartSteps for parameter ranges; document as calibration target for Phase 2.
- Risk: Archetypes too discrete; real users are heterogeneous. Mitigation: Document as limitation; plan continuous parameter sampling for Phase 2.

**Nature-publication contribution:** User archetypes enable systematic evaluation across population segments. The 4-archetype model is a simplification that must be validated against real data in Phase 2.

**Source sub-plan:** subphase_1c_user_simulation.md, docs/design/initial_design.tex §3.2 (4 archetypes)

---

### M-05: Agent Library ✅
**Objective:** Four bandit agents (Thompson Sampling, epsilon-greedy, UCB, random).

**Deliverables (completed):**
- `agents/thompson_sampling.py` — Beta-Bernoulli bandit with configurable alpha/beta priors, posterior sampling, conjugate update
- `agents/epsilon_greedy.py` — Q-learning bandit with incremental average and configurable epsilon
- `agents/ucb.py` — Upper Confidence Bound with configurable exploration constant c
- `agents/random.py` — Uniform random action selection
- All agents: string-keyed action interface (no Action enum), configurable seed via `derive_agent_seed()`
- Agent validation in schemas: type-specific required fields enforced

**Out of scope:** State-conditioned agents, contextual bandits, deep RL (DQN, PPO) — planned for Phase 2.

**Dependencies:** M-02

---

### M-06: Experiment Runner & CLI ✅
**Objective:** Config-driven experiment execution and CLI.

**Deliverables (completed):**
- `experiment.py` with `run_episode()` (single agent, CSV output) and `run_experiment()` (all agents from config, returns dict)
- `__main__.py` with `--config`, `--agent`, `--output`, `--seed`, `--verbose` flags
- Agent config sourced from YAML by default (CLI `--agent` overrides)
- Regression test: `tests/test_regression_mvp.py` with JSON fixture at seed 42
- `tests/fixtures/mvp_expected_rewards.json` — generated, committed, exact-match to 6dp

**Out of scope:** ExperimentResult with config snapshots, results directory, multi-user parallelisation — deferred to Phase 2.

**Dependencies:** M-01, M-05

---

### M-07: Synthetic Data Pipeline
**Objective:** Complete synthetic data generation with realistic wearable data distributions parameterised from published population statistics.

**Deliverables:**
- `data/synthetic.py` updated with multi-feature generation (steps, HR, sleep, sedentary time)
- Population statistics from NHANES, All of Us, UK Biobank documented in config
- Distribution parameters (mean, variance, correlations) configurable
- Synthetic data validation tests (non-negative steps, realistic ranges, no NaNs)
- Example synthetic dataset for testing

**Definition of Done:** SyntheticDataGenerator produces multi-feature data with statistical properties matching published population stats; step counts non-negative; heart rate in physiological range; sleep hours realistic; all parameters configurable via DataConfig.

**Dependencies:** M-01 (DataConfig defines generation parameters)

**Risks:**
- Risk: Synthetic data too simplistic; doesn't capture temporal correlations. Mitigation: Start with i.i.d. sampling; document as limitation; plan time-series generation for Phase 2.
- Risk: Population statistics not representative. Mitigation: Cite sources; document demographic coverage; plan stratified sampling for Phase 2.

**Nature-publication contribution:** Synthetic data enables pipeline development and agent benchmarking before real data access. The generation parameters must be documented for reproducibility.

**Source sub-plan:** subphase_1a_data_layer.md (SyntheticDataGenerator), initial_design.tex §5 (Data Sources)

---

### M-08: Evaluation Framework
**Objective:** Define and implement baselines, metrics, and statistical analysis plan for rigorous agent comparison.

**Deliverables:**
- Baseline agents: random policy, fixed policy, rule-based policy
- Metrics: cumulative regret, average reward, adherence rate, sustained behaviour change
- Statistical analysis: bootstrap CIs for pairwise comparisons, effect sizes
- Power analysis document (sample size justification)
- Evaluation tests (baselines produce expected performance, metrics computed correctly)

**Definition of Done:** 3+ baselines implemented; all metrics computed for each agent; bootstrap CIs for pairwise comparisons; power analysis documents minimum detectable effect size; statistical tests appropriate for multiple comparisons.

**Dependencies:** M-06 (experiment runner executes evaluations)

**Risks:**
- Risk: Baselines too weak; RL agent appears better than it is. Mitigation: Include strong rule-based baselines from literature.
- Risk: Statistical tests inappropriate for non-normal rewards. Mitigation: Use non-parametric tests (bootstrap, permutation) where appropriate.

**Nature-publication contribution:** Rigorous evaluation is essential for Nature Methods. The evaluation framework must meet statistical reporting standards (pre-specified analysis, effect sizes, confidence intervals).

**Source sub-plan:** Derived from Phase 1 gap analysis (no evaluation methodology in design doc)

---

### M-09: Documentation & Examples
**Objective:** Complete API documentation, example configs, and contributor guide to enable external use.

**Deliverables:**
- Sphinx or mkdocs setup with auto-generated API docs from docstrings
- 3 example experiment configs: synthetic_baseline.yml, multi_agent_comparison.yml, sensitivity_analysis.yml
- README updated with architecture diagram, usage examples, and working quickstart
- CONTRIBUTING.md updated with API documentation guide
- CHANGELOG.md created with versioning strategy

**Definition of Done:** API docs cover all public classes and functions; 3 example configs demonstrate different use cases; README quickstart runs a working experiment; architecture diagram shows data flow; new contributor can understand system from docs alone.

**Dependencies:** M-06 (experiment runner must work for examples to be meaningful)

**Risks:**
- Risk: Documentation becomes stale as code evolves. Mitigation: Add docstring checks to CI; require docstrings for public APIs.
- Risk: Example configs too complex for beginners. Mitigation: Start with minimal example; add advanced examples progressively.

**Nature-publication contribution:** Complete documentation enables reproducibility, which is a Nature requirement. Example configs demonstrate the framework's usability.

**Source sub-plan:** Derived from Phase 2 doc-alignment audit (missing docs identified)

---

### M-10: Safety & Ethics Review
**Objective:** Add safety constraints, privacy documentation, and ethics considerations for health intervention system.

**Deliverables:**
- Hard safety constraints: maximum intervention frequency, minimum recovery periods, burden threshold limits
- Safety violation logging and reporting
- Privacy documentation: GDPR/HIPAA compliance strategy, data anonymisation procedures
- Ethics section in initial_design.tex: IRB/ethics discussion, consent tracking, data use agreements
- Safety tests (constraints enforced, violations logged)

**Definition of Done:** Hard safety constraints configurable and enforced; safety violations logged with timestamps; privacy documentation covers all Phase 2 datasets; ethics section added to design doc; IRB/ethics approval status documented for real data use.

**Dependencies:** M-03 (transition/reward models), M-04 (user simulation with burden)

**Risks:**
- Risk: Safety constraints too restrictive; limit agent performance. Mitigation: Make constraints configurable; document trade-offs.
- Risk: Privacy documentation incomplete for Phase 2 datasets. Mitigation: Consult with data providers early; document data use agreements.

**Nature-publication contribution:** Health interventions require explicit safety and ethics discussion. Nature Methods requires ethical approval documentation for any human-subjects research, even simulated.

**Source sub-plan:** Derived from Phase 1 gap analysis (no safety/ethics in design doc)

## Completed Milestones

M-01, M-02, M-03, M-05, and M-06 are shipped in PR #103 (`feat/issue-101-mvp-simulator`). The remaining milestones (M-04, M-07, M-08, M-09, M-10) represent the forward roadmap.

## Success Metrics (Remaining Milestones)

### Foundation (M-07)
- **M-07:** Synthetic data statistical properties within 10% of published population statistics for steps, HR, sleep

### Core MDP (M-04)
- **M-04:** 4 archetypes produce statistically distinct response distributions (p<0.01, ANOVA)

### Integration (M-08, M-09)
- **M-08:** ≥3 baselines implemented; statistical power ≥80% for detecting 10% improvement over best baseline
- **M-09:** API docs cover 100% of public APIs; 3 example configs tested and working; README quickstart produces results in <2 minutes

### Safety (M-10)
- **M-10:** 100% of safety constraints enforced; zero safety violations in test suite; privacy documentation covers all 3 Phase 2 datasets

### Overall Success Criteria
- **Publication readiness:** All Nature Methods statistical reporting requirements met (pre-specified analysis, effect sizes, CIs)
- **Framework usability:** External researcher can configure and run experiment using only documentation
- **Reproducibility:** Same config + seed produces identical results on different machines
- **Performance:** Thompson Sampling outperforms random baseline by ≥15% cumulative reward on synthetic data
