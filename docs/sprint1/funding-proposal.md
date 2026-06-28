---
title: "Funding Proposal — Sprint 1: Reinforcement Learning for Just-in-Time Adaptive Physical Activity Interventions"
status: "draft"
date: "2026-06-28"
---

# Funding Proposal

## Reinforcement Learning for Just-in-Time Adaptive Physical Activity Interventions: A Configurable Simulation Framework

**Principal Investigator:** [To be completed]
**Institution:** [To be completed]

## Abstract

Just-in-time adaptive interventions (JITAIs) delivered via mobile devices can increase physical activity at population scale, but personalising _when_ and _how_ to intervene remains an open challenge. Reinforcement learning (RL) offers a principled framework for learning personalised intervention policies from sequential data, yet progress is hampered by the absence of standardised, reproducible simulation environments for development and evaluation. We propose to build **rl-health-interventions**, an open-source, configurable simulation framework that combines a 36-state factored Markov decision process (MDP)—capturing step count, sleep, day type, and notification burden—with large language model (LLM)-bootstrapped transition dynamics. The framework will support contextual bandit and tabular RL agents, a YAML-driven experiment configuration system, and a rigorous evaluation protocol with 50 random seeds, bootstrap confidence intervals, and pairwise hypothesis testing. The only marginal cost is approximately **$0.15 in LLM API calls** for transition table generation (DeepSeek V4 Flash). All other computation runs on local hardware. The resulting environment will enable rapid, reproducible RL research for digital health interventions at near-zero marginal cost per experiment.

## 1. Specific Aims

**Aim 1: Design and implement a 36-state factored MDP environment for physical activity JITAIs.** The state space spans four dimensions—step count bin (3 levels), sleep quality (2 levels), day type (2 levels), and notification burden (3 levels)—yielding 36 distinct states. Four actions (idle, movement suggestion, goal reminder, journal) are available at each of 5 daily decision points over 90-day episodes (450 steps total). The transition model is factored into within-day (5 tables) and day-boundary (1 table) components, with all 6 tables bootstrapped from LLM-generated data (24,480 API calls, est. $0.15–$0.33). A uniform-Dirichlet random fallback enables development without API access.

**Aim 2: Evaluate model-free RL agents against standard baselines.** Four contextual bandit algorithms (Thompson Sampling, epsilon-greedy, UCB, decaying epsilon-greedy) and optional Q-learning will be compared against random, idle-only, and fixed-ratio baselines across 50 random seeds per configuration. Primary metric is total reward over 450 steps; secondary metrics include per-step reward, convergence in the final 50 steps, and percentage gap from optimal policy. Statistical inference uses 95% bootstrap confidence intervals with Bonferroni-Holm correction.

**Aim 3: Establish a reproducible evaluation pipeline.** The YAML-driven config system ensures that identical configuration files and transition tables produce identical results across platforms. Regression test fixtures (JSON, 6 decimal places) enforce exact reproducibility. All code, configs, and generated data will be released under an open-source license.

## 2. Significance

Physical inactivity is the fourth-leading risk factor for global mortality (WHO 2020). Smartphone-based JITAIs can deliver behavioural interventions at low marginal cost and at moments when users are most receptive, potentially reaching populations with limited access to in-person health coaching. However, the complexity of personalising intervention timing and content has limited real-world effectiveness.

RL is a natural fit for JITAI personalisation: the decision problem is inherently sequential (actions affect future states through burden and behaviour change), partially observable to the user, and requires balancing exploration of new policies with exploitation of known effective strategies. HeartSteps V2 (Liao et al. 2019) demonstrated that a contextual bandit can increase step counts by 34% over random intervention scheduling in a 90-day clinical trial. Trella et al. (2022) confirmed the viability of RL-designed digital interventions across multiple deployments.

Despite these successes, RL for digital health lacks standardised simulation environments analogous to Gymnasium (Brockman et al. 2016) in robotics or Atari (Bellemare et al. 2013) in deep RL. Health Gym (Gateno et al. 2023) addresses physiological simulators but does not model notification burden, day-of-week effects, or multi-timestep daily structure—all critical for JITAI realism. StepCountJITAI (Karine et al. 2024) incorporates burden but uses a simpler state space and manual transition specification.

Our proposal addresses this gap by providing an open, configurable simulation framework with three novel features: (1) **factored transitions** generated by LLMs at negligible cost, replacing manually crafted probability tables, (2) **burden-as-state** with a rolling window that captures saturation effects without tuning parameters, and (3) **YAML-driven reproducibility** enabling exact replication across research groups without shared code pipelines.

## 3. Innovation

**LLM-bootstrapped transition models.** Prior JITAI simulators rely on hand-crafted probability tables, imposing high development cost ($5,000–$20,000 in researcher time) and limiting exploration of alternative transition structures. We use LLMs to generate realistic step count and sleep outcomes conditioned on the full state vector, then aggregate into empirical probability tables. Total API cost is $0.15 (75% cache hit rate on DeepSeek V4 Flash)—a 10,000× reduction in marginal cost per experiment. Any researcher can generate new transition tables for alternative assumptions (e.g., different populations, intervention intensities) in hours instead of weeks.

**Burden as rolling-window state.** Existing burden models require manual specification of decay functions or fatigue accumulation rates with no empirical basis. Our rolling-window count (non-idle actions in the last 3 timesteps) has zero tuning parameters while capturing the clinically meaningful distinction between low, medium, and high notification burden. The window crosses day boundaries, ensuring burden does not reset unrealistically at midnight.

**Factored day-boundary transitions.** Most JITAI simulators either ignore sleep or model it as an exogenous variable. Our day-boundary transition table explicitly models how end-of-day state (step total, burden level) affects sleep quality, which in turn modulates next-day activity probability. Sleep is a well-established moderator of physical activity (Irwin 2016), and including it as an endogenous state variable allows RL agents to learn policies that consider long-term consequences of intervention decisions.

**YAML-driven experiment definition.** All MDP parameters, reward functions, agent configurations, and evaluation settings are specified in a single YAML file. No code changes are required to modify state cardinalities, action sets, reward weights, or transition models. This enables non-programmer researchers (e.g., behavioural scientists) to design and run experiments through configuration files alone, lowering the barrier to RL adoption in digital health.

## 4. Approach

### 4.1 MDP Formalisation

The environment is a finite-horizon Markov decision process $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ with: full observability, fixed 450-step episodes (90 days × 5 steps/day), no discounting for bandits ($\gamma = 1.0$), immediate reward horizon (30-minute window per HeartSteps V2), and a single blank context.

**State space.** $s_t = (\text{step\_bin}, \text{sleep}, \text{day\_of\_week}, \text{burden})$.

Step bin: inactive (<800 steps/timestep, daily <4k following Lee 2022), moderate (800–1,600, daily 4k–8k, Saint-Maurice 2020), active (>1,600, daily >8k, Paluch 2022). The LLM outputs raw step counts; the environment accumulates into daily totals and bins. Sleep: rested (>7h) or under-rested (≤7h), updated at day boundary. Day of week: weekday or weekend. Burden: low (0 non-idle in last 3 steps), medium (1), high (2 or 3). Rolling window crosses day boundaries; first timestep defaults to low. Cardinality: 3 × 2 × 2 × 3 = **36 states**.

**Action space.** Four actions: idle (no intervention), movement_suggestion (recommend a walk), goal_reminder (remind of step goal), journal (expressive writing, no step benefit). The set size is consistent with Trella 2022 (which prescribes 2–6 actions, each mapping to a distinct behavioural construct). All non-idle actions carry equal burden impact. Journal serves as a burden control: same cost, no activity benefit.

**Transition model.** Factored into within-day (5 tables, step 1–4, step_bin only) and day-boundary (step 0, sleep). All 6 tables LLM-bootstrapped: 24,480 calls across 864 input pairs, 10 samples per output cell. Uniform-Dirichlet random fallback for development without API access.

**Reward.** $R = f(\text{step\_bin}') - \lambda \cdot \mathbb{1}[\text{action} \neq \text{idle}]$, where $f$ maps step bins to scalar rewards and $\lambda$ is the action penalty multiplier — both specified in the YAML config (see Section 4.4).

**Agent suite.** Multiple baselines (random, idle-only, fixed-ratio) and bandit algorithms (Thompson Sampling, epsilon-greedy, UCB, decaying epsilon-greedy) with optional Q-learning for non-myopic comparison. Each bandit maintains independent (s,a) parameter groups.

### 4.2 LLM Bootstrapping Protocol

**Model selection.** DeepSeek V4 Flash via OpenRouter ($0.09/1M input tokens, $0.18/1M output tokens). At 75% cache hit rate, estimated total cost is $0.15 (uncached: $0.33). This is 40–70× cheaper than GPT-5.5 or Claude Opus 4.8 ($7.40–$7.80 estimated).

**Prompt design.** A system prompt (cached once) defines the 5-timestep daily structure, step ranges, sleep threshold, and burden levels. Within-day prompts condition on time of day, current state, and the action taken (or "No action" for idle). Day-boundary prompts summarise the full day's steps and notifications, then ask for a sleep duration prediction. All prompts request a single structured JSON output per call.

**Pipeline.** For each of 864 (s, a) pairs: render the prompt, call the LLM, parse the output, bin the result into the appropriate step or sleep category. Repeat 10× per output cell (24,480 total calls). Aggregate empirical frequencies into probability tables. Write 6 JSON files (5 within-day + 1 day-boundary). The full pipeline runs in approximately 4–6 hours with concurrent API calls.

### 4.3 Evaluation Protocol

**Metrics.** Total reward ($\sum R_t$), per-step reward ($\frac{1}{450} \sum R_t$), convergence (mean of last 50 steps), and percentage gap vs optimal policy.

**Protocol.** 50 random seeds per configuration. Reporting: mean ± 95% bootstrap CI. Pairwise bootstrap test against best baseline with Cohen's d effect size. Bonferroni-Holm correction across all pairwise comparisons.

**Verification checks.** (1) Same config + seed = identical CSV on any platform. (2) idle-only yields lower total reward than random. (3) High burden reduces probability of step_bin improvement relative to low burden. (4) Thompson Sampling achieves ≥5% reward gain over random baseline.

### 4.4 Implementation Architecture

**Config file.** Single YAML file specifies: initial state, state dimensions (names, boundaries), actions with penalties, reward function weights, transition model type (bootstrap or random), episode length, steps per day, and random seed.

**Key interfaces.** StateView (frozen dataclass with state_key property → tuple), TransitionModel (ABC with `transition(state, action) → StateView`), RewardHandler (ABC with `reward(state, action, next_state, step_idx) → float`), Agent (ABC with `select_action(state) → str` and `update(state, action, reward, next_state)`).

**Step logic.** Step 0: sample day-boundary transition → sleep'. Steps 1–4: sample within-day table → step_bin'. Day advances every 5 steps. Burden recalculated from rolling window. Reward computed from next state and action.

**Reproducibility.** Fixed seed per config with derived sub-seeds per agent. CSV output with columns: step, day, step_of_day, step_bin, sleep, day_of_week, burden, action, reward. Regression tests with JSON fixtures (exact match, 6 decimal places).

## 5. Resource Requirements

### 5.1 Compute

The only paid computation is the LLM bootstrap: 24,480 calls to DeepSeek V4 Flash via OpenRouter at approximately **$0.15** total (75% prompt cache hit rate). All experiment runs (50 seeds × 6 agents × 450 steps) execute on a standard laptop — no GPU or cloud compute required.

### 5.2 Timeline

| Month | Milestone | Deliverable |
|-------|-----------|-------------|
| 1–2 | Core MDP environment | StateView, TransitionModel, RewardHandler, step logic |
| 2–3 | LLM bootstrapping | Pipeline script, 6 transition table JSON files |
| 3–4 | Agent implementations | Baselines + bandit agents, extendable agent API |
| 4–5 | Evaluation pipeline | CSV output, bootstrap CI, hypothesis tests, 50-seed runner |
| 5–6 | Validation | Verification checks, regression fixtures, cross-platform testing |
| 6–7 | Documentation & release | README, example configs, tutorial notebook, PyPI release |
| 7–8 | Sensitivity analysis | λ sweep, burden window size, seed sensitivity |
| 9–10 | Paper preparation | Manuscript, figures, code repository freeze |
| 11–12 | Submission & revision | Journal submission, response to reviewers |

## 6. Dissemination Plan

- **Open-source release:** MIT license, GitHub repository, PyPI package, documentation site.
- **Journal publication:** Target _JMIR mHealth and uHealth_ or _Scientific Reports_.
- **Conference presentation:** _Machine Learning for Health (ML4H)_ or _AMIA Annual Symposium_.
- **Tutorial materials:** Jupyter notebook demonstrating config-driven experiments, reproducible from the published repository.
- **Generated data:** All 6 transition tables and all 50-seed experiment outputs released alongside code.

## 7. Decision Summary

| Decision | Resolution | Evidence |
|----------|-----------|----------|
| Step bins | 3: <4k / 4k–8k / >8k daily | Lee 2022, Saint-Maurice 2020, Paluch 2022 |
| Factored state | Within-day + day-boundary tables | Trella 2022, HeartSteps V2 |
| Psychosocial state | Sleep in; mood/stress out | Rabbi 2019 (null), Irwin 2016 |
| Trend | Excluded | No published precedent |
| Time-of-day | Implicit table index (5 tables) | Liao 2019 |
| Day type | Weekday/weekend as state dimension | All 6 reference systems |
| Action set | idle, movement, goal, journal | Sprint 1 design (informed by Trella 2022 framework, HeartSteps V2) |
| Non-activity reward | Deferred | No burden-reduction evidence |
| Mood as reward | Deferred | No deployed RL system does this |
| Burden model | Rolling window, 3 levels | StepCountJITAI, zero tuning params |
| Reward function | step_bin reward − λ·𝟙[action≠idle] | Sprint 1 design (informed by Trella 2022 framework, StepCountJITAI) |
| Algorithm | Model-free bandits + Q-learning | HeartSteps V2 (34% increase) |
| Evaluation | 4 metrics, 3 baselines, 50 seeds | MVP experimental protocol |

## 8. References

1. Klasnja P, et al. (2019). HeartSteps V1 MRT. _Annals of Behavioral Medicine_, 53(11), 989–998.
2. Liao P, et al. (2019). HeartSteps V2 Contextual Bandits. _JMIR mHealth and uHealth_, 7(8), e10419.
3. Trella AL, et al. (2022). Designing RL for Digital Interventions. _NPJ Digital Medicine_, 5, 142.
4. Karine RN, et al. (2024). StepCountJITAI. _JMIR Serious Games_, 12, e58363.
5. Lee IM, et al. (2022). Step Volume and Mortality. _JAMA Internal Medicine_, 182(6), 624–632.
6. Saint-Maurice PF, et al. (2020). Daily Steps and Mortality. _JAMA_, 323(12), 1151–1160.
7. Paluch AE, et al. (2022). Daily Steps and All-Cause Mortality. _The Lancet Public Health_, 7(2), e114–e125.
8. Gateno D, et al. (2023). Health Gym. _NeurIPS Datasets and Benchmarks_.
9. Smyth JM, et al. (2018). Expressive Writing and Health. _JMIR Mental Health_, 5(3), e10207.
10. Irwin MR, et al. (2016). Sleep Interventions. _The Lancet_, 387(10032), 1603–1611.
11. Rabbi M, et al. (2019). Depression MRT: Stress as Moderator. _JMIR mHealth and uHealth_, 7(6), e12254.
12. Sutton RS, Barto AG. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press.
13. Auer P, et al. (2002). Multiarmed Bandit Analysis. _Machine Learning_, 47, 235–256.
14. Watkins CJCH, Dayan P. (1992). Q-Learning. _Machine Learning_, 8, 279–292.

## 9. Dependencies

Python 3.11+, numpy, polars, pydantic, pyyaml, httpx (LLM script only), pytest, ruff, ty. Managed via uv.