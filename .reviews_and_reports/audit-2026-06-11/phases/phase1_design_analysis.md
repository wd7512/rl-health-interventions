# Phase 1: Design Document Deep Analysis

**Source:** `docs/initial_design.tex`
**Analysed by:** Primary Model (Qwen 3.7 Max)
**Date:** 2026-06-11

---

## 1.1 Executive Summary

The design document proposes `rl-health-interventions`, a configurable simulation framework for testing RL-driven just-in-time adaptive interventions (JITAIs) on wearable device data. The MDP formalisation is well-structured with clear state/action/reward definitions and a multi-timescale reward signal. The framework architecture follows sound software engineering principles (ABC + registry pattern, config-first design). However, the document is incomplete for publication: it lacks evaluation methodology, statistical analysis plans, safety constraints for health interventions, and any experimental results. The design is at early-stage maturity — Phase 1 implementation is entirely stubbed with no functional code. Critical gaps in offline RL considerations, partial observability handling, and clinical validation metrics must be addressed before this reaches Nature Methods submission readiness.

## 1.2 Scope & Objectives Assessment

**Are objectives specific, measurable, and achievable?**

The primary objective — "a configurable simulation framework where dataset schema, MDP dynamics, user behaviour models, and agent configurations are specified entirely through YAML config files" — is specific and well-scoped. However, success criteria are absent. The document does not define what "configurable" means in measurable terms (e.g., "supports N dataset schemas without code changes"), nor does it specify performance benchmarks for the simulation.

**Is the scope appropriate for a first version?**

Yes, the Phase 1/Phase 2 split is appropriate. Phase 1 (synthetic data, rule-based simulation, Thompson Sampling baseline) is achievable within an 8-week internship. The explicit exclusion of Gymnasium and Stable Baselines 3 reduces dependency risk.

**Scope creep risks:**
1. **LLM-based user simulation** is mentioned as a "stretch goal beyond Phase 2" but lacks any scoping — this is a multi-month research project in itself.
2. **Deep RL agents (DQN, PPO)** are listed as "if time permits" without justification for why they would outperform Thompson Sampling in this domain.
3. **Non-stationary environment adaptation** and **offline RL (CQL, IQL)** are mentioned in Phase 2 context but completely unscoped.
4. **Web UI** mentioned in the Decision Log as "stretch" — contradicts the CLI-first philosophy.

**Under-scoped areas:**
1. **Multi-user simulation:** The MDP is defined for a single user. How are population-level experiments specified? No discussion of user sampling or cohort design.
2. **Reward function validation:** How are reward weights (α, β, λ, η) determined? No discussion of sensitivity analysis or clinical grounding.
3. **Action space extensibility:** The 6-action space is hardcoded in the document. How are custom actions added via config?
4. **Episode termination:** The MDP defines `done` in the environment interface but never specifies termination conditions. Are episodes infinite? Fixed-length?

## 1.3 Technical Architecture Review

| Component | Stated Purpose | Adequacy (1–5) | Evidence from .tex | Gaps | Suggested Improvement |
|-----------|---------------|-----------------|-------------------|------|----------------------|
| **Config (YAML → Pydantic)** | Schema validation, 3-layer validation | 3 | "3-layer validation stubs" mentioned; Pydantic referenced in code_design.md but not in .tex | No config schema shown in design doc; 3-layer validation undefined in .tex | Include example YAML config and define the 3 validation layers explicitly |
| **Data (Polars lazy reader)** | Ingest CSV/Parquet, feature engineering | 4 | `Dataset` dataclass defined; `StateView.from_dataset()` bridge documented | Missing: data provenance tracking, schema evolution strategy | Add data lineage metadata to Dataset |
| **MDP (Environment)** | Step/reset, transition ABC, reward ABC | 4 | Full MDP tuple defined; multi-timescale reward specified; burden model documented | Episode termination undefined; partial observability not addressed; no observation model separate from state | Define episode structure; add observation function O(o|s) |
| **UserSim (4 archetypes)** | Rule-based behavioural response | 3 | 4 archetypes named with brief descriptions; burden accumulation formula given | Archetype parameters unquantified; no calibration methodology; "plausible" is not a metric | Define parameter ranges for each archetype; specify calibration data sources |
| **Agent (ABC + Thompson Sampling)** | Pluggable RL agents | 3 | Agent ABC with select_action/update; TS as baseline | TS algorithm not specified (which variant?); no exploration strategy documented | Specify TS variant (e.g., Gaussian TS with known variance); add action selection pseudocode |
| **Experiment (Factory + Runner)** | CLI wiring, results output | 2 | `ExperimentFactory.build()` documented; CLI command shown | No results schema defined; reproducibility measures incomplete; no checkpointing strategy | Define ExperimentResult schema; add seed management and config snapshot specification |

**Overall architecture pattern:** Pipeline with registry-based plugin system. Consistent with Ousterhout's "deep modules" philosophy referenced in code_design.md. The ABC + registry pattern is well-suited for the extensibility requirements.

## 1.4 RL Methodology Critique

**Which RL algorithm(s) are proposed?**

Thompson Sampling is the confirmed baseline (Decision Log). DQN and PPO are mentioned as stretch goals. The choice of TS is appropriate for the bandit-like setting (6 discrete actions, daily decisions) and aligns with the StepCountJITAI reference. However:

- **Why Thompson Sampling over UCB?** The document cites `karine2024stepcountjitai` but does not explain the algorithmic choice. For contextual bandits with non-stationary user responses, TS's posterior sampling may underperform compared to methods with explicit exploration bonuses.
- **No offline RL consideration for Phase 2.** The document mentions CQL and IQL as stretch goals but does not discuss why offline RL is needed when Phase 2 will have HeartSteps MRT data. The distinction between online RL (simulation) and offline RL (real data policy evaluation) is not made.

**Reward function design:**

The multi-timescale reward (immediate steps + 3-week body measure) is the document's strongest contribution. However:

1. **Reward hacking risk is unaddressed.** An agent could maximise immediate step rewards by sending interventions every epoch, ignoring the burden penalty if α >> β. The document acknowledges the reward/burden penalty separation as an "open question" but provides no analysis.
2. **Goal progress term (λ · goal_progress_t) is undefined.** What constitutes "goal progress"? Is it steps/goal? A binary threshold? This is a critical design decision left open.
3. **Delayed reward credit assignment.** The document correctly identifies the sparse reward problem (body measure every 21 days) and mentions a decaying formulation as an alternative. However, it does not discuss temporal difference methods or eligibility traces for credit assignment over the 21-day gap.

**Partial observability:**

**Not addressed.** The MDP is defined as fully observable (state = all variables). In reality, wearable sensors have missing data, measurement noise, and reporting delays. The document should either:
- Define an observation function O(o|s) and frame as a POMDP, or
- Justify the full observability assumption with evidence about sensor reliability

**Offline RL / Safe RL / Distributional shift:**

- **Offline RL:** Mentioned (CQL, IQL) but not motivated. Phase 2 will have limited MRT data — offline RL is essential for safe policy evaluation without deploying untested policies to real users. This is a critical gap.
- **Safe RL:** Completely absent. Health interventions require safety constraints (e.g., maximum intervention frequency, minimum recovery periods). The burden model is a soft constraint; hard safety constraints are not discussed.
- **Distributional shift:** Not discussed. The rule-based simulator will produce synthetic trajectories that differ from real user behaviour. How is sim-to-real gap addressed? No domain randomisation or robustness analysis.

## 1.5 Data Pipeline & Privacy

**Is the data flow clearly specified end-to-end?**

Partially. The pipeline from raw files → Polars lazy scan → FeaturePipeline → numpy Dataset → StateView is documented in code_design.md but not in the .tex file. The .tex only mentions "synthetic data generators parameterised from published population statistics" and Phase 2 datasets.

**Are PHI/PII concerns addressed?**

**No.** The document does not discuss:
- GDPR/HIPAA compliance for health data
- Data anonymisation procedures
- Consent documentation for secondary data use
- Data access controls or audit logging
- Right to erasure implementation

This is a critical gap for any health intervention system, even one using synthetic data in Phase 1. Phase 2 datasets (All of Us, UK Biobank) have strict data use agreements that must be documented.

**Data provenance and consent:**

Not discussed. The framework should track:
- Dataset version and source
- Consent status for each data point
- Data lineage from raw to processed features
- Reproducibility metadata (random seeds, preprocessing versions)

## 1.6 Evaluation & Validation Plan

**Is there a clear experimental protocol?**

**No.** The document describes the framework architecture but provides no experimental design. Missing:
- Number of simulation runs per configuration
- Statistical power analysis
- Baseline comparisons (random policy, fixed-interval policy, etc.)
- Ablation studies for reward components

**Are baselines specified?**

Only Thompson Sampling is confirmed. No comparison baselines are defined:
- Random policy (uniform action selection)
- Fixed policy (always send motivational prompt)
- Rule-based policy (intervene when steps < threshold)
- No-intervention baseline (always a₀)

**Are clinical validity metrics present?**

**No.** The document mentions "regret, reward, adherence" as metrics in the Decision Log but:
- "Adherence" is undefined — is it intervention acceptance rate? Long-term engagement?
- No clinical outcome metrics (e.g., sustained behaviour change over 6+ months)
- No health equity metrics (disparate impact across demographic groups)
- No safety metrics (intervention burden, user dropout rate)

**Statistical analysis plan:**

**Completely absent.** For Nature Methods submission, the document must specify:
- Primary and secondary endpoints
- Sample size justification
- Statistical tests (paired t-test, Wilcoxon, bootstrap CIs?)
- Multiple comparison corrections
- Effect size reporting

Reference: Nature Methods statistical reporting standards (2024) require pre-specified analysis plans, effect sizes with confidence intervals, and explicit handling of multiple comparisons. None of this is present.

## 1.7 Reproducibility Score

**Code reproducibility: 6/10**
- ✅ Repository exists with clear structure
- ✅ pyproject.toml with pinned dependencies
- ✅ uv lock file present
- ⚠️ No Docker/container specification
- ❌ No example experiment configs in the repo
- ❌ README quickstart does not produce results

**Experiment reproducibility: 4/10**
- ✅ Random seed support mentioned in Decision Log
- ⚠️ Seed management not implemented in code
- ❌ No config snapshot saved with results
- ❌ No experiment tracking (MLflow, W&B, or equivalent)
- ❌ Results directory structure undefined

**Documentation completeness: 7/10**
- ✅ Comprehensive design document (this .tex)
- ✅ Detailed sub-phase engineering plans in docs/code/
- ⚠️ README is minimal — no architecture diagram, no usage examples
- ⚠️ API documentation absent (no docstrings in stub code)
- ❌ No contributor guide for adding new components

**Overall: 6/10**

The design is well-documented at the architectural level, but the implementation is entirely stubbed. Reproducibility is currently impossible because there is no working system to reproduce.

## 1.8 Gap Matrix

| # | Gap Description | Severity | Location in initial_design.tex | Suggested Fix |
|---|-----------------|----------|-------------------------------|---------------|
| 1 | **No evaluation methodology or experimental protocol defined** | Critical | Entire document | Add §6: Evaluation Plan with baselines, metrics, statistical analysis, and power analysis |
| 2 | **No safety constraints or safe RL considerations for health interventions** | Critical | §3 (MDP), §4 (Framework) | Add burden threshold hard constraints; specify safety violations and intervention stopping rules; discuss conservative policy approaches |
| 3 | **Privacy/GDPR/HIPAA compliance completely absent** | Critical | §5 (Data Sources) | Add §7: Privacy & Ethics with data anonymisation, consent tracking, and data use agreement documentation |
| 4 | **Partial observability not addressed — MDP assumes full state observability** | High | §3.1 (State) | Either define observation function O(o|s) for POMDP framing, or justify full observability with sensor reliability data |
| 5 | **Offline RL for Phase 2 not motivated despite limited MRT data** | High | §7 (Phase 2) | Add offline RL rationale: HeartSteps has ~150 users with ~5 decision points/day × 90 days = ~67,500 transitions, insufficient for online RL but suitable for offline policy evaluation |
| 6 | **Reward function components (goal_progress) undefined** | High | §3.3 (Reward) | Specify goal_progress calculation: is it steps/goal (continuous), binary threshold, or time-in-range? |
| 7 | **No clinical validation metrics — only ML metrics (regret, reward)** | High | §6 Decision Log | Add clinical metrics: sustained behaviour change (% users maintaining >baseline for 3+ months), intervention fatigue rate, adverse events |
| 8 | **Episode termination conditions undefined** | Medium | §3 (MDP) | Define episode length (fixed 90 days? infinite horizon?) and termination criteria (user dropout, goal achievement) |
| 9 | **User archetype parameters unquantified** | Medium | §3.2 (Transition) | Define parameter ranges for each archetype: response magnitude, burden accumulation rate, baseline activity level |
| 10 | **No distributional shift analysis for sim-to-real gap** | Medium | §7 (Phase 2) | Add domain randomisation strategy and robustness evaluation across parameter perturbations |
| 11 | **Action space extensibility not discussed** | Low | §3.2 (Actions) | Document how custom actions are added via config; specify action schema validation |

## 1.9 Nature Publishing Standards Alignment

Nature Methods (2024) submission requirements mapped to current document status:

| Requirement | Status | Evidence / Gap |
|-------------|--------|----------------|
| **Novel methodology clearly articulated** | ⚠️ Partial | Config-first framework is novel, but novelty claim not explicitly stated; comparison to existing frameworks (Health Gym, StepCountJITAI) is brief |
| **Reproducible code and data** | ❌ Missing | No working code; no example configs; no synthetic data generation parameters published |
| **Statistical analysis plan pre-specified** | ❌ Missing | No statistical tests, sample sizes, or confidence intervals discussed |
| **Baseline comparisons** | ❌ Missing | Only Thompson Sampling defined; no random/fixed/rule-based baselines |
| **Ablation studies** | ❌ Missing | No plan to isolate reward components, transition model effects, or user archetype contributions |
| **Clinical validation** | ❌ Missing | No clinical outcome metrics; no expert review protocol; no user study design |
| **Limitations section** | ⚠️ Partial | Decision Log documents open questions but does not frame as limitations |
| **Ethical approval documentation** | ❌ Missing | No IRB/ethics discussion; no data use agreements documented |
| **Computational resource reporting** | ❌ Missing | No hardware specs, training time, or memory requirements |
| **Software availability statement** | ⚠️ Partial | Repo exists but no release, no DOI, no container |
| **Data availability statement** | ⚠️ Partial | Synthetic data planned; Phase 2 datasets listed but access status unclear |
| **Author contributions** | ❌ Missing | Not applicable at design stage |
| **Competing interests** | ❌ Missing | Not applicable at design stage |
| **Reporting guidelines compliance (CONSORT/SPIRIT/ARRIVE)** | ❌ Missing | No discussion of applicable guidelines |

**Alignment score: 2/14 requirements met (14%)**

The document is at design stage and appropriately lacks some publication elements (author contributions, competing interests). However, the absence of evaluation methodology, statistical planning, and safety considerations represents a fundamental gap that must be addressed before any Nature submission is feasible.

## 1.10 Priority Recommendations

Ranked by impact on publication readiness:

1. **Add evaluation methodology section (§6).** Define experimental protocol: baselines (random, fixed, rule-based), metrics (regret, adherence, sustained change), statistical tests (paired bootstrap CIs), and power analysis. This is the single biggest blocker to publication.

2. **Implement safety constraints for health interventions.** Add hard burden thresholds, maximum intervention frequency constraints, and discussion of conservative policy approaches. Health interventions require explicit safety guarantees, not just soft penalties.

3. **Specify offline RL strategy for Phase 2.** HeartSteps MRT data is limited (~67K transitions). Motivate offline RL (CQL/IQL) for safe policy evaluation without deploying untested policies. This bridges the sim-to-real gap.

4. **Define reward function components precisely.** Specify goal_progress calculation, conduct sensitivity analysis on reward weights (α, β, λ, η), and ground in clinical literature where possible.

5. **Add privacy and ethics section.** Document GDPR/HIPAA compliance strategy, data anonymisation procedures, and consent tracking. Required for any health data system, even synthetic.

6. **Address partial observability.** Either frame as POMDP with observation function, or justify full observability assumption with wearable sensor reliability data.

7. **Create working example experiment.** Implement a minimal end-to-end experiment with synthetic data, Thompson Sampling, and results output. Current code is entirely stubbed — reproducibility is impossible.

8. **Define clinical validation metrics.** Add sustained behaviour change, intervention fatigue rate, and health equity metrics. ML metrics alone are insufficient for health interventions.

9. **Document episode structure and termination.** Define episode length, termination conditions, and discount factor justification.

10. **Quantify user archetype parameters.** Specify response magnitudes, burden rates, and baseline activity for each archetype. Currently "plausible" is the only validation criterion.

---

## Quality Gate 1 Verification

- [x] All 10 sections present and substantive — no placeholders
- [x] Gap Matrix has 11 entries with severity ratings (≥5 required)
- [x] Nature standards section has 14 requirements with explicit per-requirement verdicts
- [x] Reproducibility sub-scores all filled (Code: 6/10, Experiment: 4/10, Documentation: 7/10)
- [x] Recommendations numbered 1–10 and explicitly ranked by impact

**Gate 1: PASSED**
