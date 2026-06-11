# Technical Dependencies, Risk Register, TRL, and Stub Tracker

Generated: 2026-06-11
Sources: `docs/initial_design.tex`, `docs/ROADMAP.md`, `reports/phase1_design_analysis.md`, `src/` sources, `config/` sources, `pyproject.toml`

---

## Technical Dependency Graph

```mermaid
flowchart TD

    %% ===== CONFIG LAYER =====
    subgraph Config["Config Layer"]
        SCHEMA["config/schemas.py\n— MDPConfig, AgentConfig,\nExperimentConfig, ActionSpec\n— Pydantic models\nMISSING: not yet implemented"]
        LOADER["config/loader.py\n— YAML → Pydantic\n— 3-layer validation\nMISSING: not yet implemented"]
        DCONF["config/datasets/*.yaml\n— 14 dataset configs\n14 configs exist"]
    end

    %% ===== DATA LAYER =====
    subgraph Data["Data Layer (src/data/)"]
        BASE["_base.py\n— DataConfig (Pydantic)\n— DatasetProtocol"]
        DSET["dataset.py\n— Dataset dataclass"]
        POLARS["polars_reader.py\n— scan_csv/parquet\nwith timeout"]
        LOADERS["loaders.py\n— 12 dataset loaders\n— get_all_loaders()\n— load_all()"]
        SYNT["synthetic.py\n— SyntheticDataGenerator\n(only steps, stub)"]
        FEAT["feature_pipeline.py\n— FeaturePipeline.stub"]
        DATAINIT["__init__.py\n— Registry + make()"]
    end

    %% ===== MDP / CORE LAYER =====
    subgraph Core["Core MDP Layer (missing src/)"]
        STATE["state.py (StateView)\nMISSING: not yet implemented"]
        ENV["environment.py\n— Environment step/reset\nMISSING: not yet implemented"]
    end

    %% ===== TRANSITION & REWARD =====
    subgraph TR["Transition & Reward"]
        TINIT["transitions/__init__.py\n— Registry + make()"]
        TBASE["transitions/_base.py\n— TransitionModel ABC"]
        TRULE["transitions/rule_based.py\n— RuleBasedTransition\nSTUB: returns state unchanged"]

        RINIT["rewards/__init__.py\n— Registry + make()"]
        RBASE["rewards/_base.py\n— RewardHandler ABC"]
        RCOMP["rewards/compound.py\n— CompoundReward\nSTUB: returns (0.0, False)"]
    end

    %% ===== USER SIMULATION =====
    subgraph Sim["User Simulation"]
        SINIT["simulation/__init__.py\n— Registry + make()"]
        SBASE["simulation/_base.py\n— ResponseModel ABC"]
        SRULE["simulation/rule_based.py\n— RuleBasedResponse\nSTUB: returns 0.0"]
        UPROF["simulation/user_profile.py\n— UserProfile with 4 archetypes\nMISSING: not yet implemented"]
    end

    %% ===== AGENT =====
    subgraph Agent["Agent"]
        AINIT["agents/__init__.py\n— Registry + make()"]
        ABASE["agents/_base.py\n— Agent ABC"]
        ATS["agents/thompson_sampling.py\n— ThompsonSamplingAgent\nSTUB: returns action 0 always"]
    end

    %% ===== EXPERIMENT =====
    subgraph Exp["Experiment Runner (missing src/)"]
        EFACT["experiment/factory.py\n— ExperimentFactory\nMISSING: not yet implemented"]
        ERUN["experiment/runner.py\n— Experiment\nMISSING: not yet implemented"]
        MAIN["__main__.py\n— CLI entry point\nSTUB: prints hello only"]
    end

    %% ===== EVALUATION =====
    subgraph Eval["Evaluation (missing)"]
        EVALM["MISSING:\n— Baselines (random, fixed,\n  rule-based)\n— Metrics computation\n— Statistical analysis"]
    end

    %% ===== SAFETY =====
    subgraph Safe["Safety (missing)"]
        SAFEM["MISSING:\n— Hard burden thresholds\n— Intervention frequency limits\n— Safety violation logging\n— Privacy documentation"]
    end

    %% ===== DEPENDENCY EDGES =====
    %% Config depends on pydantic
    SCHEMA --> PYD
    LOADER --> SCHEMA
    LOADER --> PYD
    DCONF -.-> LOADER

    %% Data layer
    BASE --> PYD
    DSET --> NP
    POLARS --> PD
    POLARS --> REQ
    LOADERS --> POLARS
    LOADERS --> KG
    LOADERS --> DS
    LOADERS --> HF
    LOADERS --> PND
    LOADERS --> UCI
    LOADERS --> WF
    LOADERS --> REQ
    SYNT --> NP
    SYNT --> DSET
    FEAT -.-> DSET

    %% Core MDP depends on data + config
    STATE --> DSET
    ENV --> STATE
    ENV --> SCHEMA

    %% Transitions & Rewards
    TRULE --> TBASE
    TRULE --> SCHEMA
    RCOMP --> RBASE
    RCOMP --> SCHEMA
    TBASE --> STATE
    RBASE --> STATE

    %% User Simulation
    SRULE --> SBASE
    SRULE --> SCHEMA
    UPROF --> SCHEMA
    SBASE --> STATE

    %% Agent
    ATS --> ABASE
    ABASE --> STATE

    %% Experiment runner ties everything together
    EFACT --> SCHEMA
    EFACT --> ENV
    EFACT --> TRULE
    EFACT --> RCOMP
    EFACT --> UPROF
    EFACT --> SRULE
    EFACT --> ATS
    EFACT --> DSET
    ERUN --> EFACT
    MAIN --> ERUN

    %% Evaluation depends on experiment
    EVALM --> ERUN

    %% Safety depends on transition + simulation
    SAFEM --> ENV
    SAFEM --> UPROF
```
### External Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Numerical operations, array processing |
| polars | Fast CSV/Parquet data reading |
| pydantic | Config schema validation |
| datasets (HF) | HuggingFace dataset access |
| huggingface-hub | Model/dataset hub integration |
| kagglehub | Kaggle dataset download |
| pandas | Data manipulation (secondary) |
| ucimlrepo | UCI ML Repository access |
| wfdb | WFDB signal data (physiological) |
| requests | HTTP requests for data retrieval |
| Python >=3.11 | Runtime requirement |



---

## Risk Register

| Risk ID | Description | Affected Milestone(s) | Likelihood (H/M/L) | Impact (H/M/L) | Mitigation Strategy |
|---------|-------------|----------------------|-------------------|----------------|---------------------|
| R-01 | **Reward hacking** — Agent learns to maximise short-term step rewards by sending interventions every epoch, ignoring burden penalty if α ≫ β. Burden mechanism becomes ineffective. | M-03, M-05, M-06 | H | H | Implement explicit counterweight in compound reward (penalty scaling with intervention frequency); add reward normalisation; include reward-shaping tests that verify burden-sensitive behaviour; document reward weight sensitivity analysis. |
| R-02 | **Distribution shift (sim-to-real gap)** — Rule-based synthetic transitions produce unrealistic trajectories. Agent policies trained on synthetic data fail to transfer to real user behaviour in Phase 2. | M-03, M-04, M-06, M-08 | H | H | Document domain randomisation strategy for Phase 1 parameters; plan calibration against HeartSteps response data in Phase 2; implement robustness evaluation by perturbing transition parameters ±20%; add distributional shift metrics (DKL between synthetic/held-out distributions). |
| R-03 | **Safety violations** — No hard safety constraints implemented. Agent could select high-burden interventions repeatedly, causing simulated user burnout or producing clinically unsafe policies when transferred. | M-04, M-10 | H | H | Implement hard burden thresholds (B_max) in Environment.step() that reject actions exceeding threshold; add minimum recovery periods between non-zero interventions; enforce configurable max interventions-per-day; log and surface all safety violations in experiment output. |
| R-04 | **Data leakage across user profiles** — User identification vectors or combined state representations may leak information between synthetic user trajectories, biasing agent evaluation. | M-02, M-05, M-06 | M | M | Ensure per-user data separation in Dataset and StateView; add per-user random seeds in simulation; verify that agent update is strictly per-trajectory; document data isolation design in experiment runner. |
| R-05 | **Clinical validity failures** — Reward weights (α, β, λ, η) and archetype parameters are not clinically grounded. Agent may optimise proxies that don't correspond to meaningful health outcomes. | M-03, M-04, M-08 | H | H | Document all parameter sources with citations from StepCountJITAI and HeartSteps literature; plan sensitivity analysis across parameter space; add clinical metrics (sustained behaviour change, intervention fatigue) beyond ML metrics; define minimum clinically meaningful effect size before Phase 2. |
| R-06 | **Config schema rigidity** — Pydantic models for DataConfig, MDPConfig, AgentConfig, ExperimentConfig become too rigid to support future extensions (new action types, observation functions, custom reward shapes). | M-01 | M | M | Use Pydantic's `extra="allow"` for forward compatibility; document extension points in schema; define a version field for config evolution; add integration tests that verify unknown fields don't crash config loading. |
| R-07 | **Multi-timescale reward credit assignment failure** — The sparse 3-week delayed body measure reward is too distant from the actions that produced it. Agent fails to learn long-term behaviour change strategies. | M-02, M-03, M-05 | M | H | Implement both sparse and decaying delayed reward formulations (configurable); add eligibility trace diagnostics; verify that Thompson Sampling posterior update captures delayed effects; document decaying reward as recommended alternative in user-facing docs. |
| R-08 | **Missing datasets block Phase 2 validation** — Access to All of Us (controlled tier, 4-8 week application) and HeartSteps data (requires author permission) may be delayed or denied, preventing MDP calibration. | M-06, M-08, M-09 | H | H | Begin data access applications as early as possible (Week 1); maintain synthetic-only validation pipeline as fallback; document data access status in README; plan calibration against published summary statistics if raw data is unavailable. |
| R-09 | **User archetype discretisation fails to capture heterogeneity** — Four discrete archetypes (goal-driven, social, resistant, stable) cannot span the real population distribution, making simulator evaluations misleading. | M-04 | M | H | Document discrete archetypes as a known limitation; plan continuous-parameter sampling for Phase 2; validate archetype separation with ANOVA tests (success metric: p<0.01); add mixture-of-archetypes capability as stretch goal. |
| R-10 | **Partial observability unaddressed** — The MDP assumes full state observability but real wearable data has missing sensor readings, measurement noise, and reporting delays, invalidating MDP assumptions for real-data runs. | M-02, M-03, M-06 | M | H | Frame as POMDP from the start with an observation function O(o|s) in Environment; add missing-data masks to StateView; implement observation noise injection in synthetic pipeline; document POMDP vs MDP assumptions in design doc. |
| R-11 | **Evaluation metrics insufficient for publication** — Only regret/reward metrics computed without clinical validation, statistical confidence intervals, or comparison baselines, failing Nature Methods requirements. | M-08 | H | H | Implement ≥3 baselines (random, fixed, rule-based) as blocking criterion for M-08; compute bootstrap CIs for all pairwise comparisons; add pre-registered analysis plan documented in repo; include effect sizes and power analysis. |
| R-12 | **Experiment runner performance bottleneck** — Single-threaded experiment loop with 100 users × 90 days × multiple agents may take >1 hour, hindering iterative development and large-scale parameter sweeps. | M-06 | M | M | Profile experiment loop early in development; use NumPy vectorisation for batch operations; implement optional multi-user parallelisation with `concurrent.futures`; target <5 min for 100 users × 90 days. |
| R-13 | **Dependency version conflicts** — The dependency set (numpy, polars, pandas, pydantic, datasets, kagglehub, ucimlrepo, wfdb) may have conflicting constraints as packages evolve. | M-01, M-06, M-07 | L | M | Pin exact versions in `uv.lock`; run CI with `uv sync --frozen` daily; add dependency conflict check to CI pipeline; upgrade one dependency at a time with full test suite run. |

---

## Technology Readiness Levels (TRL)

TRL definitions (adapted for computational research):  
1 — Basic principles observed / concept formulated  
2 — Technology concept / architecture specified  
3 — Analytical/experimental proof of concept  
4 — Component validated in laboratory (unit tests pass)  
5 — Component validated in relevant environment (integration tests pass)  
6 — System demonstrated in relevant environment (end-to-end runs)  
7 — System demonstrated in operational environment  
8 — System complete and qualified  
9 — Actual system proven in operational environment  

| Component | Current TRL | Target TRL | Gap Description | Blocking Milestone |
|-----------|-------------|------------|-----------------|-------------------|
| **Config Schema & Validation** (`config/schemas.py`, `config/loader.py`) | 1 | 7 | No config schemas exist except DataConfig in `data/_base.py`. Missing: MDPConfig, AgentConfig, ExperimentConfig, ActionSpec, RewardWeights. No YAML loader, no 3-layer validation pipeline. 14 dataset YAML files exist but are not consumed by any code. | M-01 |
| **StateView & Environment** (`src/rl_health_interventions/state.py`, `environment.py`) | 1 | 6 | Both files are completely missing. StateView must be a dataclass with `from_dataset()` classmethod. Environment must implement `step()` / `reset()` with multi-timescale reward. Environment calls transition+reward models. | M-02 |
| **Transition Models** (`transitions/`) | 2 | 7 | TransitionModel ABC exists with abstract `transition()`. RuleBasedTransition registered but returns state unchanged — no behavioural response logic, no burden accumulation, no archetype parameterisation. | M-03 |
| **Reward Models** (`rewards/`) | 2 | 7 | RewardHandler ABC exists with abstract `reward()`. CompoundReward registered but returns `(0.0, False)` — no immediate reward computation, no delayed body-measure reward, no configurable weights. | M-03 |
| **User Simulation** (`simulation/`) | 2 | 7 | ResponseModel ABC exists with abstract `response()`. RuleBasedResponse returns 0.0 — no archetype logic, no UserProfile class, no parameter ranges for 4 archetypes. | M-04 |
| **Thompson Sampling Agent** (`agents/thompson_sampling.py`) | 2 | 7 | Agent ABC exists. ThompsonSamplingAgent always returns action 0 — no Gaussian/Beta posterior, no prior configuration, no posterior update. Can't demonstrate regret decrease. | M-05 |
| **Experiment Runner & CLI** (`experiment/factory.py`, `runner.py`, `__main__.py`) | 1 | 7 | Experiment directory does not exist. `__main__.py` prints "Hello" only — no CLI parsing, no config loading, no experiment loop, no results output. | M-06 |
| **Synthetic Data Pipeline** (`data/synthetic.py`) | 3 | 6 | SyntheticDataGenerator exists and produces Dataset objects with steps, but only uses univariate normal — no multi-feature generation (HR, sleep, sedentary), no configurable distribution parameters, no temporal correlations. FeaturePipeline is a stub. | M-07 |
| **Data Loaders** (`data/loaders.py`) | 5 | 6 | 12 dataset loaders implemented with download, caching, error handling, and standardised output. Missing: `allofus_fitbit` loader (BigQuery, requires AoU workbench access), `stepcountjitai` loader (no code integration yet). Bulk `load_all()` works. | M-07 |
| **Evaluation Framework** (not yet created) | 1 | 6 | No baselines (random, fixed, rule-based). No metrics computation (regret, reward, adherence). No statistical analysis (bootstrap CIs, effect sizes, power analysis). No evaluation module exists. | M-08 |
| **Documentation & Examples** (`docs/`, `README.md`) | 3 | 7 | Design doc exists (.tex) and is comprehensive. Architecture sub-plans exist. README is minimal — no architecture diagram, no quickstart that produces results. No API docs. No example configs for actual experiments. | M-09 |
| **Safety & Ethics** (not yet created) | 1 | 6 | No hard safety constraints in any component. No burden threshold enforcement. No privacy documentation. No ethics review or IRB discussion. Maximum intervention frequency not configurable. | M-10 |

---

## Stub-to-Implementation Tracker

| Stub File | What It Must Implement | Test Criteria | Blocking Milestone ID |
|-----------|-----------------------|---------------|----------------------|
| `src/rl_health_interventions/data/synthetic.py` — `SyntheticDataGenerator.generate()` | Generate multi-feature synthetic wearable data (steps, HR, sleep hours, sedentary minutes) with configurable distribution parameters (mean, variance, correlations). Support NHANES-calibrated population statistics. Include temporal correlation across timesteps. | • Output `Dataset` object validates successfully<br>• Step counts non-negative, HR in [30,220], sleep hours in [0,24]<br>• Feature means within 10% of configured population parameters over 1000 samples<br>• No NaN/Inf values<br>• Seeded reproducibility (same seed → identical data) | M-07 |
| `src/rl_health_interventions/data/feature_pipeline.py` — `FeaturePipeline.from_config()` | Parse config dict to build transformation chain: column selection → normalisation → feature engineering (time-of-day encoding, day-of-week encoding, rolling averages). Support composable transforms registered via the ABC+registry pattern. | • Pipeline produces correct output shapes per transform<br>• Normalisation maps to [0,1] range<br>• Config validation rejects invalid transforms with clear error<br>• Empty config produces identity pipeline | M-07 |
| `src/rl_health_interventions/transitions/rule_based.py` — `RuleBasedTransition.transition()` | Implement behavioural response model: compute Δsteps based on action and user profile archetype. Implement burden accumulation/decay (linear accumulator with configurable decay rate δ and max threshold B_max). Support 4 archetypes with distinct response parameters. Return new state. | • Goal-driven archetype responds more to reminders/feedback than other archetypes<br>• Burden increases on intervention, decays on no-action (a₀)<br>• Burden > B_max triggers linear response decay<br>• Configurable parameters (α_Δsteps, burden_decay, B_max) affect output as expected<br>• All 4 archetypes produce statistically distinct response distributions (p<0.01, ANOVA) | M-03 |
| `src/rl_health_interventions/rewards/compound.py` — `CompoundReward.reward()` | Compute immediate reward: R_immediate = α·Δactivity - β·burden_penalty + λ·goal_progress. Compute delayed reward every 21 epochs: R_delayed = η·BM_improvement. Return (reward, done) tuple. Support configurable weights (α, β, λ, η) and action penalties. | • Immediate reward computed correctly for each action with configured penalties<br>• Delayed reward non-zero only on epochs t ≡ 0 (mod 21)<br>• Goal_progress term reflects steps/goal ratio<br>• Config weight changes produce proportional reward changes<br>• Reward stays within expected bounds | M-03 |
| `src/rl_health_interventions/simulation/rule_based.py` — `RuleBasedResponse.response()` | Implement archetype-specific response magnitude given (state, action, profile). 4 archetypes: goal-driven (responds to reminders/feedback), social responder (responds to motivational prompts), resistant (low response, fast burden accumulation), stable maintainer (already active, low marginal gain). Return response magnitude. | • Goal-driven: response to a₂ (walking suggestion) and a₃ (goal reminder) > response to a₁<br>• Social: response to a₁ (motivational prompt) > response to a₂/a₃<br>• Resistant: overall response magnitude < 30% of other archetypes; burden saturates 2× faster<br>• Stable maintainer: high baseline, low marginal gain from any action<br>• All response values in [0, 1] | M-04 |
| Not yet created — must implement `simulation/user_profile.py`: `UserProfile` | Pydantic schema with fields: archetype (Literal enum), baseline_activity (low/med/high), response_params (dict of action→response params), burden_params (decay rate, max threshold). Factory method to instantiate pre-defined archetype profiles. | • 4 pre-defined archetype profiles produce correct parameter sets<br>• Serialisation round-trips via JSON<br>• Validation rejects invalid archetype names with clear error<br>• All parameters have sensible defaults documented | M-04 |
| `src/rl_health_interventions/agents/thompson_sampling.py` — `ThompsonSamplingAgent` | Implement Gaussian Thompson Sampling with known variance (or Beta TS for binary rewards). Configurable prior parameters (μ₀, σ²₀). `select_action()` samples from posterior and returns argmax. `update()` performs conjugate Bayesian update. Support exploration temperature parameter. | • `select_action()` returns a valid action index (0–5)<br>• Posterior mean converges to true action value over 1000 steps<br>• Regret decreases ≥20% over 1000 episodes on known-optimal bandit problem<br>• Prior parameters affect initial exploration behaviour<br>• Multiple calls with same state produce varied actions due to posterior sampling | M-05 |
| `src/rl_health_interventions/__main__.py` — `main()` | Parse CLI arguments (--config CONFIG_PATH, --seed SEED, --output OUTPUT_DIR). Load config from YAML → Pydantic via ExperimentFactory. Run Experiment. Output results: console table + CSV/JSON + config snapshot. Exit with informative messages on errors. | • `uv run rl-health-interventions --config experiments/demo.yml` runs end-to-end<br>• CLI --help shows all flags with descriptions<br>• Invalid config produces actionable ValidationError<br>• Same config + seed produces identical results across 2 runs<br>• Results directory contains CSV + YAML sidecar | M-06 |
| Not yet created — must implement `experiment/factory.py`: `ExperimentFactory` | Build experiment components from config: instantiate Dataset from DataConfig, Environment from MDPConfig, Agent from AgentConfig, UserProfiles from profile config. Wire them into Experiment loop. Validate component compatibility (dummy step). | • Build succeeds for valid config<br>• Build fails with clear error for incompatible components (e.g., agent expects different state space)<br>• Registry lookup works for all registered components<br>• Dummy step catches wiring errors before full run | M-06 |
| Not yet created — must implement `experiment/runner.py`: `Experiment` | Run trial loop for N users × T timesteps: for each user, reset Environment, loop: agent selects action → environment steps → reward computed → agent updates. Accumulate metrics per user/per epoch. Return ExperimentResult with config snapshot, seeds, metrics. | • Experiment completes without errors for valid config<br>• Per-user metrics computed (cumulative reward, adherence, regret)<br>• Config snapshot saved in results directory<br>• Random seeds produce deterministic results<br>• Multi-user simulation correctly isolates user trajectories | M-06 |
| Not yet created — must implement `config/schemas.py` | Pydantic models: DataConfig (file_path, format, column_mapping), MDPConfig (state_vars, action_specs, reward_weights, discount_factor, episode_length), ActionSpec (action_id, label, burden_penalty), RewardWeights (alpha, beta, lambda, eta), AgentConfig (type, prior_params, hyperparams), ExperimentConfig (dataset, mdp, agent, user_profiles, n_users, n_timesteps, seed). | • All config types load correctly from valid YAML<br>• Invalid configs raise ValidationError with actionable messages<br>• Default values work when optional fields omitted<br>• Nested model validation catches type errors<br>• Forward-compatible with extra fields (extra="allow") | M-01 |
| Not yet created — must implement `config/loader.py` | 3-layer validation pipeline: Layer 1 (schema validation via Pydantic models), Layer 2 (registry validation — verify component names exist in registries), Layer 3 (dummy step — instantiate components with dummy data and verify they produce correct types). Return validated ExperimentConfig. | • All 3 layers execute in sequence<br>• Layer 1 rejects malformed YAML<br>• Layer 2 rejects unknown registry component names<br>• Layer 3 catches shape/type mismatches between components<br>• Clear error messages identify which layer failed and why | M-01 |
| Not yet created — must implement `src/rl_health_interventions/state.py`: `StateView` | Dataclass with fields matching MDP state variables: steps_t, hr_t, sleep_hours_t, sedentary_min_t, time_of_day_t, day_of_week_t, goal_progress_t, burden_t, prev_action, prev_response, body_measure_k, age, gender, baseline_activity. `from_dataset(dataset, user_idx, t)` classmethod. Normalisation to [0,1]. | • All fields present with correct types<br>• `from_dataset()` extracts correct slice for user/timestep<br>• Normalised values in [0,1] range<br>• Missing features raise clear error<br>• Supports extension via dict[str, float] fallback | M-02 |
| Not yet created — must implement `src/rl_health_interventions/environment.py`: `Environment` | `Environment` class with `__init__(config, transition_model, reward_handler)`, `step(state, action) → (StateView, float, bool)`, `reset() → StateView`. Multi-timescale reward accumulator. Episode termination logic (fixed-length default 90 days). Safety constraint enforcement (burden threshold, max intervention frequency). | • `step()` returns correctly typed tuple<br>• Reward contains both immediate and delayed components at correct epochs<br>• Episode termination works at configured length or via early stop<br>• Safety constraints enforced (actions rejected if over threshold)<br>• `reset()` returns initial state<br>• Multiple episodes produce independent trajectories | M-02 |

**Note on stub identification:** Every concrete implementation file under `src/rl_health_interventions/` that inherits from an ABC (or is an entry point) is a stub — none perform their intended computation. The 4 ABC files (`_base.py` under each module) define correct interfaces. 4 major modules are entirely missing (state.py, environment.py, experiment/, config/schemas.py, config/loader.py). `__main__.py` is a stub (prints "Hello" only). The test suite only has registry and import tests — no functional tests exist for any component.
