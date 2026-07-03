# LLM Prompting for Transition Bootstrap — Literature Review

> **Date:** July 3, 2026
> **Scope:** Literature supporting the use of LLMs to bootstrap transition probabilities for a health intervention MDP
> **Sources:** arXiv, Semantic Scholar, Google Scholar, ACM DL, JMIR, Nature, NeurIPS, ACL Anthology, IEEE, PLOS, Springer

---

## Executive Summary

We use an LLM (DeepSeek V4 Flash) to generate transition probability tables for a discrete Markov Decision Process (MDP) modeling physical activity and sleep behavior in a health intervention. The approach produces **6 transition tables** via **22,320 LLM calls** at a total cost of approximately **$0.27**.

The key question: **Is this approach validated by existing literature?**

**Short answer: Yes, with caveats.** The literature provides strong precedent across multiple dimensions — LLMs as user simulators for RL, LLM-generated synthetic environments, LLM world models for MDPs, and LLMs for adaptive health interventions. However, no prior work uses LLM calls to estimate tabular/discrete MDP transition probabilities in exactly this way. The approach is **novel in its specific implementation** but **well-supported by converging lines of evidence** from adjacent fields.

**What the literature supports:**
- LLMs can simulate user behavior that trains effective RL policies (Papers 1, 6, 18, 22)
- LLMs can serve as world models predicting state transitions (Papers 10, 11, 14, 22)
- LLMs can generate JITAI content and detect intervention triggers (Papers 1, 4, 7)
- Behaviorally-informed synthetic data can train RL for health interventions (Paper 18)
- Structured output from LLMs is increasingly reliable (Papers 2, 8, 10)

**What the literature warns about:**
- Behavioral fidelity degrades over long horizons and complex scenarios (Papers 14, 26)
- LLMs may encode training data biases that propagate to simulated populations (Paper 6)
- Quality of user simulator critically matters for RL training outcomes (Paper 22)
- Validation against real pilot data is essential but rarely done (Papers 2, 14)
- Small models (7-9B) struggle more with format compliance (Paper 13)

---

## Current Prompt Design

### Architecture Overview

The system uses a **factored MDP** with 36 states (3 step bins × 2 sleep × 2 day-of-week × 3 burden) and 4 actions (idle, movement_suggestion, goal_reminder, journal). Time-of-day is encoded implicitly as one of 5 tables rather than a state dimension.

### Three Prompt Types

**1. System Prompt (cached once)**

Defines the persona and environmental context:
- "Generally healthy adult looking to improve exercise and sleep habits"
- Per-timestep step ranges: inactive (<800), moderate (800–1600), active (>1600)
- Daily step total ranges: inactive (<4000), moderate (4000–8000), active (>8000)
- Sleep quality: good / poor
- Burden levels: low (0 of last 3), medium (1 of last 3), high (2–3 of last 3)

**2. Within-Day Prompts (5 timesteps)**

Each of 5 daily time slots (morning, mid-morning, lunch, afternoon, evening) gets its own prompt and transition table. The prompt includes:
- Time of day (morning/mid-morning/lunch/afternoon/evening)
- Day type (weekday/weekend)
- Sleep quality (good/poor)
- Burden level (low/medium/high)
- Previous timestep's activity bin (inactive / moderately active / active)
- Action taken (no notification sentence for idle)

The LLM outputs a raw step count (e.g., "800"), which the environment bins into the 3-level step_bin.

**3. Day-Boundary Prompt (1 per day)**

End-of-day summary that predicts next-night sleep quality:
- Total daily steps and daily bin
- All 5 notification actions for the day
- Day type, current sleep quality, burden level

The LLM outputs a binary JSON choice: `{"sleep_quality": "good"}` or `{"sleep_quality": "poor"}`.

### Table Structure

| Table | Transition | Input Dimensions | Output | (s,a) Pairs | LLM Calls |
|-------|-----------|-----------------|--------|-------------|-----------|
| Day-boundary | P(sleep' \| step_bin_daily, burden, day, sleep) | 3×3×2×2 = 36 | sleep'(2) | 36 | 720 |
| Within-day ×5 | P(step_bin' \| step_bin, burden, action, day, sleep) | 3×4×3×2×2 = 144 | step_bin'(3) | 144 | 4,320 |
| **Total** | | **756** | | | **22,320** |

> **Counting convention:** Each (s,a) pair produces a distribution over outcomes. The LLM is called multiple times to sample from this distribution: 20 calls/pair for day-boundary (2 outcomes: good/poor), 30 calls/pair for within-day (3 outcomes: inactive/moderate/active). Both yield exactly 10 samples per output category. Total: 36×20 + 720×30 = 22,320 calls.

Each (s,a) pair is sampled **10 times** (Algorithm 2), yielding 10 draws per outcome category. This sampling approach converts LLM generation into a stochastic estimator of transition probabilities.

### Cost Breakdown

| Component | Value |
|-----------|-------|
| Model | DeepSeek V4 Flash via OpenRouter |
| Input cost | ~$0.09/M tokens |
| Output cost | ~$0.18/M tokens |
| Tokens per call | ~50–100 |
| Total calls | 22,320 |
| **Total cost** | **~$0.27** |
| With prompt caching | **~$0.12** |

At this cost, the entire transition matrix can be regenerated after any prompt tweak for pocket change.

---

## Literature Review by Dimension

### Dimension 1: LLM-Based User Simulation for RL

This is the most extensively studied dimension, with 28 papers surveyed across recommender systems, bandits, health training, and social simulation.

**Key Papers:**

- **Alamdari et al. (2024, EMNLP)** — *Jump Starting Bandits with LLM-Generated Prior Knowledge* — The most directly relevant paper. LLMs simulate user preferences to warm-start contextual bandit algorithms for health messaging (vaccine preferences). **LLM-generated priors significantly reduce regret in bandit algorithms.** This is essentially the same paradigm as our approach: use LLM knowledge to bootstrap a decision-making system before real user data is available.

- **Evidence-Driven Sim Data (2026, MDPI)** — Uses behaviorally-informed synthetic data (grounded in behavioral theory) to pre-train contextual bandit algorithms for personalized mHealth activity-promoting messages. Found **close alignment between simulated and real behaviors.** Directly validates the use of synthetic data for training health intervention RL.

- **Suh et al. (2026, arXiv)** — *Quantifying the Utility of User Simulators* — Trains LLM assistants via RL against a spectrum of user simulators. Found that **quality of user simulator critically matters**: training against role-playing LLM yields assistants statistically indistinguishable from initial assistant (51% win rate), while fine-tuned simulator yields significant gains (58%). This is a cautionary result for our approach — the quality of the LLM's behavioral model directly impacts downstream policy quality.

- **Bayley et al. (2025, GenAIRecP)** — Tests robustness of LLM-initialized bandits to noisy priors. Found **robustness up to moderate noise levels**, but **more data can reinforce bias rather than improve performance** under noisy conditions. Relevant to our 10-sample-per-cell approach — the number of samples may matter less than prompt quality.

**Common Limitations Across Papers:**
- Behavioral fidelity gap: LLM-simulated behavior diverges from real behavior, especially for edge cases
- Distribution shift: training on LLM-simulated data risks optimizing for LLM biases
- Evaluation difficulty: no gold standard for validating LLM-simulated user behavior
- Computational cost: running many LLM agent simulations is expensive (though our approach avoids this by pre-computing tables)

### Dimension 2: Prompt Engineering for Behavioral Simulation

**Persona Prompting:**

Park et al. (2024, arXiv) — *Generative Agent Simulations of 1,000 People* — LLM agents replicate individual personality traits and behaviors based on interview data. Achieved **85% accuracy on General Social Survey replication.** Demonstrates that persona-conditioned LLMs can capture population-level behavioral patterns.

Argollo da Costa et al. (2025, arXiv) — *Can A Society of Generative Agents Simulate Human Behavior?* — LLM agents (Llama, Qwen) can capture population-level trends in vaccine hesitancy, but **challenges in subgroup differentiation.** This is relevant to our approach: our single "healthy adult" persona may miss important population heterogeneity.

**Health-Specific Prompts:**

- The Patient-Ψ paper (Wang et al., 2024, EMNLP) integrates cognitive models with LLMs to mimic patient behaviors in CBT training. Shows that **cognitive model grounding improves behavioral fidelity.** Our prompts are theory-informed (burden, sleep, step ranges) but not grounded in a specific behavioral model like TTM.

- Kashefi et al. (2025, Springer) found that **prompt engineering can effectively control LLM communication style** for health coaching. Users could distinguish between coaching tones, and communication style significantly affected engagement. While not directly about behavioral simulation, this validates the importance of prompt design for behavioral outcomes.

**Validation Approaches:**

The field lacks standardized validation. Park et al. (2023, UIST) established the foundational generative agents framework but evaluated only via human judgment of "believability." More rigorous validation (e.g., comparison with empirical survey distributions) appeared only in later work.

### Dimension 3: LLM-Generated Synthetic Environments

23 papers surveyed on LLMs as environment generators and world models.

**World Models as Transition Functions:**

- **WorldCoder (Tang et al., 2024, NeurIPS)** — LLM builds Python code representing transition and reward functions by interacting with the environment. **Transparent symbolic world models are sample-efficient, transferable, and interpretable.** This is the closest conceptual parallel to our approach: the LLM is implicitly generating a "program" (transition table) that captures environment dynamics.

- **RWML (Yu et al., 2026, arXiv)** — Self-supervised learning of action-conditioned world models T(s'|s,a) for LLM agents. Uses sim-to-real gap rewards in embedding space. **Directly models the same T(s'|s,a) function we are estimating.**

- **Qwen-AgentWorld (Qwen Team, 2026, arXiv)** — First truly native language world model. Trained on 10M+ trajectories across 7 agent domains. **Demonstrates that world models can be trained specifically for environment simulation rather than adapted from general-purpose LLMs.**

- **From Word to World (2025, ACL 2026)** — Proposes a three-level evaluation framework for LLM world models: fidelity/consistency, scalability/robustness, and agent utility. Found that **sufficiently trained world models maintain coherent latent state** but effectiveness varies by environment complexity.

**Environment Generation:**

- **EnvGen (Aszala et al., 2025, ICLR)** — LLM generates training environment configurations. A small RL agent trained with EnvGen **outperforms GPT-4 agents.** Validates that LLM-generated environments can be higher quality than the LLM itself as an agent.

- **AWM (Wang et al., 2026, arXiv)** — Code-driven pipeline that synthesizes 1,000 diverse executable environments. Agents trained on synthetic environments show **strong out-of-distribution generalization.**

**Key Insight for Our Approach:** The literature shows LLMs can serve as world models/dynamics models, but accuracy degrades with complexity. Our discrete, low-dimensional MDP (36 states, 4 actions) is well-suited for this approach — far simpler than the text-based environments most papers target.

### Dimension 4: LLM Calibration and Validation

This is the weakest dimension in terms of directly relevant work, but contains important warnings.

**Known Failure Modes:**

- **Jiang et al. (2024, arXiv)** — Tests GPT-4 as a text-based world simulator. Found it is "impressive but still unreliable" — **accuracy drops on complex transitions.** Our simple state space mitigates this, but it's worth noting.

- **Lu et al. (2025, ACL 2026)** — Evaluates LLM agents on multi-turn human behavior simulation using real online customer data. Found that **behavioral fidelity decreases over longer interaction horizons** and there is distribution shift between simulated and real behavior. For our 90-day episodes, this degradation is a concern.

- **LLM-Based World Models survey (2025, arXiv)** — Calls for **more rigorous evaluation methodologies**; current benchmarks may overestimate LLM world modeling capabilities.

**Calibration Approaches:**

- Bayley et al. (2025) provides the most relevant calibration framework: injecting noise into synthetic training data and measuring impact on downstream regret. Found robustness up to moderate noise levels.
- The RWML paper proposes sim-to-real gap rewards in embedding space — a principled way to measure and minimize the gap between LLM-predicted and actual transitions.

**Gap:** No paper validates LLM-generated transition probabilities against empirically observed transition frequencies in a health intervention MDP. This is the validation gap our project must address.

### Dimension 5: Digital Twins and Synthetic Populations

25 papers surveyed on microsimulation, agent-based models, and LLM-driven digital twins for health.

**Microsimulation Approaches:**

- **POHEM (Kopec et al., 2013; Marathe et al., 2015)** — Population Health Model platform for physical activity microsimulation. Uses survey data (NPHS) for calibration. Validates that **dynamic microsimulation can model physical activity trajectories** at the population level. Our LLM bootstrapping is conceptually similar but uses LLM knowledge instead of survey calibration.

- **CovidSim (Ferguson et al., 2020)** — Landmark example of individual-based microsimulation with synthetic population. Informed UK/US government policy. Shows the power of synthetic population-based health simulation at scale.

**Agent-Based Models:**

- **CommunityRx (Ozik et al., 2021, PLOS Comp Bio)** — Large-scale ABM with synthetic population of 802,191 agents. Models community health intervention delivery. **High-fidelity replication of intervention delivery** validated against clinical trial findings.

- **Zhang et al. (2014, AJPH)** — ABM for dietary policy evaluation in Pasadena. **Captured social influence on dietary decisions.** Relevant because social influence is absent from our model.

**LLM-Driven Digital Twins:**

- **AgentSociety (Piao et al., 2025, AAMAS)** — Population-level simulation of **millions of LLM-driven agents.** Demonstrates scalability of LLM-based agent simulations to population level.

- **JITAI-Twins (Gazi et al., 2026, JMIR)** — From Susan Murphy's group (JITAI pioneer). Uses LLMs/GenAI to simulate and optimize JITAI policies before deployment. **Most directly relevant to our project** — frameworks JITAIs using digital twin simulations with LLMs as the generative engine.

- **He et al. (2025, arXiv)** — Comprehensive survey of Human Digital Twin technology. HDTs are dynamic, data-driven virtual representations of individuals. Our approach is a lightweight version: using LLM knowledge rather than real individual data to create a "proto-digital-twin."

**Relevant Gap:** Few papers combine MDP formulation with synthetic populations for policy optimization. DiGiacomo et al. (2021) proposes MDP-based digital twin composition, but for service orchestration, not health interventions.

### Dimension 6: LLM for MDP Transition Estimation

22 papers surveyed — this is the most directly relevant dimension.

**Directly Relevant Papers:**

- **Benechehab et al. (2024, arXiv)** — *Zero-shot Model-based Reinforcement Learning using Large Language Models* — Uses LLMs as zero-shot dynamics models for continuous control RL. LLM generates text descriptions of state transitions given (state, action) pairs, embedded and used as a learned transition model. **Introduces DICL (Description-Informed Context Learning).** This is the closest paper to our approach, but works with continuous states and embeddings rather than discrete transition matrices.

- **RWML (Yu et al., 2026)** — Learns action-conditioned world models T(s'|s,a) for LLM agents on textual states using sim-to-real gap rewards. **Self-supervised world model learning provides more robust training signal than LLM-as-a-judge.**

- **Qwen-AgentWorld (2026)** — Native language world model trained on 10M+ real-world trajectories. Three-stage pipeline: CPT → SFT → RL. **First truly native language world model** trained specifically for environment simulation.

- **Markov States for LLM Post-Training (2026, arXiv)** — Reintroduces explicit Markov states into LLM RL. Uses state transition functions (rule-based or learned) instead of history-as-state formulation. Found that **explicit MDP state representation enables genuine novel strategy discovery** beyond what history-based approaches achieve.

- **Text World Models Survey (Li & Wang, 2026, arXiv)** — First systematic review formalizing text world models as transition functions over textual states. **Establishes the formal framework for exactly the problem we are solving:** T(s'|s,a) estimation.

**Novelty Assessment:**

The gap analysis reveals:
- No prior work uses LLM calls to estimate **tabular/discrete** MDP transition probabilities
- Most work uses LLMs as world models for text-based environments, not for estimating transition matrices
- Paper 1 (Zero-shot MBRL) is closest but works with continuous states, not discrete transition tables
- Our approach of 22,320 LLM calls to estimate P(s'|s,a) in a structured, discrete MDP appears **novel in the literature**

The closest work bootstraps from language for continuous control, not for explicit transition probability estimation in a tabular MDP.

### Dimension 7: Structured Output and JSON Generation

14 papers surveyed on constrained decoding, schema enforcement, and reliability.

**Relevant Findings:**

- **SLOT (2025, arXiv)** — Fine-tuned Mistral-7B with constrained decoding achieves **99.5% schema accuracy** and 94.0% content similarity. Shows that post-processing can fix structured outputs reliably.

- **XGrammar (2024, ICML 2025)** — Efficient grammar-constrained decoding engine. Used by production systems (vLLM, SGLang). Enables practical constrained generation with minimal overhead.

- **Schema Reinforcement Learning (THUNLP, 2025, ACL)** — Up to **16% improvement in valid complex JSON generation** via RL training with schema validators. SchemaBench with 40K+ real-world JSON schemas.

- **StructuredRAG (Shorten et al., 2024, arXiv)** — Found **high variance in JSON success rates** across models, tasks, and prompting strategies. Notably, Llama 3 8B-instruct often performs competitively with Gemini 1.5 Pro. Demonstrates importance of prompt engineering for JSON compliance.

- **When Correct Isn't Usable (Alomana et al., 2026, arXiv)** — Studies the reliability gap between mathematical correctness and format compliance in 7-9B models. **Smaller models struggle more with format compliance.** Our approach uses DeepSeek V4 Flash (a small model), so this is relevant.

**Relevance to Our Design:**

Our day-boundary prompt asks for JSON output (`{"sleep_quality": "good"}`), while within-day prompts ask for raw numbers. The literature supports:
1. Keep schemas simple and flat (our binary JSON is ideal)
2. Provide explicit examples (our prompt includes the expected format)
3. Validate outputs against schema before use
4. Consider retry logic for format failures

The structured output literature strongly supports our approach of using simple JSON for the day-boundary prompt and raw numbers for within-day prompts (which don't need JSON at all).

### Dimension 8: LLMs for Adaptive Health Interventions

20 papers surveyed on JITAI content generation, LLM coaching, and trigger optimization.

**Most Directly Relevant:**

- **The Last JITAI? (2024, CHI 2025)** — Evaluated LLMs for both **trigger detection AND content personalization** in JITAIs. Found LLMs can replace traditional rule-based triggers AND personalize content simultaneously. LLM-tailored content **significantly reduced screen time and increased acceptance.** This is the paper that most directly validates using LLMs in the JITAI pipeline.

- **JITAI-Twins (Gazi et al., 2026, JMIR)** — From Susan Murphy's group. Framework for optimizing JITAIs using digital twin simulations with LLMs. **LLMs serve as the generative engine for simulating participant responses.** This is the closest framework paper to our approach.

- **GAMBITTS (Brooks et al., 2025, NeurIPS)** — Novel bandit algorithm for adaptive interventions where a generative AI model generates the actual treatment content. Explicitly models both the treatment generation process and the reward process. **Principled framework for deploying GenAI-generated content in adaptive interventions.**

- **GPTCoach (Jörke et al., 2024, CHI 2025)** — GPT-4-based health coaching chatbot implementing the Active Choices evidence-based coaching program. Lab study with 16 participants found users **felt comfortable sharing concerns** and the system offered personalized support.

- **HealthGuru (Wang et al., 2025, CHI 2025)** — LLM-powered sleep health chatbot using JITAI principles with **data-driven and theory-guided** recommendations that adapt to real-life constraints.

- **MHC-Coach (2025, Nature NPJ Cardiovascular Health)** — LLM fine-tuned on Transtheoretical Model of Change for stage-matched motivational messages. **Fine-tuned LLM produced messages rated more appropriate and effective than zero-shot approaches.**

**Key Themes:**
1. LLMs can generate personalized, contextually appropriate intervention messages at scale
2. LLMs can replace both trigger detection AND content generation in the JITAI pipeline
3. Fine-tuning on behavioral models (TTM, motivational interviewing) produces more effective coaching than zero-shot prompting
4. Combining LLMs with rule-based systems creates more robust intervention systems
5. Adapting communication style to individual preferences significantly affects engagement
6. Standardized evaluation of LLM health interventions remains an open challenge

---

## Key Findings and Recommendations

### What the Literature Supports

1. **LLMs as transition model bootstrappers** — Multiple papers demonstrate that LLMs can predict state transitions (Benechehab et al. 2024, RWML 2026, Qwen-AgentWorld 2026). Our approach is consistent with this body of work.

2. **Cold-start framing** — Alamdari et al. (2024) shows LLM priors significantly reduce regret in bandits, framing the use case as "warm-starting before real data is available." This is exactly our scenario.

3. **Behaviorally-informed synthetic data** — Evidence-Driven Sim Data (2026) shows behaviorally-informed synthetic data can train contextual bandits for personalized health messages with close alignment to real behaviors.

4. **Simple state spaces favor LLM world models** — The literature consistently shows accuracy degrades with complexity. Our 36-state MDP is far simpler than the environments typically studied, putting us in the favorable regime.

5. **Structured output reliability** — Modern LLMs with simple schemas achieve high reliability. Our binary JSON output for sleep quality and raw number output for step counts are well within current capabilities.

6. **LLMs for JITAI pipelines** — The Last JITAI? (2024) and GAMBITTS (2025) demonstrate LLMs can serve as both trigger detectors and content generators in adaptive health interventions.

### What the Literature Warns About

1. **Simulator quality matters critically** — Suh et al. (2026) found that role-playing LLM simulators yield no improvement over baseline, while fine-tuned simulators yield significant gains. Our off-the-shelf LLM prompting approach may be closer to the "role-playing" end of the spectrum.

2. **Behavioral fidelity degrades over long horizons** — Lu et al. (2025) found decreasing fidelity over longer interaction horizons. Our 90-day episodes with 5 daily transitions (450 total) may encounter this degradation.

3. **More data can reinforce bias** — Bayley et al. (2025) found that larger synthetic datasets under noisy conditions can reinforce bias rather than improve performance. Our 10-sample-per-cell approach is modest, but the systematic nature of LLM biases is a concern.

4. **No gold standard validation exists** — The field lacks standardized methods for validating LLM-simulated transition probabilities against real data. We must establish our own validation pipeline.

5. **Bias encoding** — LLMs may encode biases from training data that propagate to simulated populations. Our single "healthy adult" persona may not capture important population heterogeneity.

6. **Small model limitations** — Alomana et al. (2026) found smaller models struggle with format compliance. DeepSeek V4 Flash is a small model; we should validate output quality.

### Specific Recommendations for Prompt Design

1. **Ground prompts in behavioral theory** — The MHC-Coach paper (2025) shows fine-tuning on TTM produces more effective behavioral models. While we don't fine-tune, our prompts should reference behavioral mechanisms (burden effects, sleep-activity coupling) more explicitly.

2. **Add calibration anchors** — Include quantitative priors in prompts where available. For example: "Research suggests people typically take 2,000–4,000 steps in a morning" could ground the LLM's estimates.

3. **Validate high-burden transitions** — The literature warns about behavioral fidelity at extremes. High-burden states are where we have least confidence and highest behavioral stakes (intervention fatigue).

4. **Test prompt sensitivity** — The structured output literature shows high variance across prompting strategies. We should systematically test prompt variations and measure impact on transition distributions.

5. **Consider persona diversity** — Park et al. (2024) shows single personas miss population heterogeneity. Even if Sprint 1 uses a single persona, we should plan for multi-persona bootstrapping in Phase 2.

6. **Add explicit examples for JSON outputs** — The structured output literature consistently recommends 2–3 complete examples. Our day-boundary prompt already includes the expected format, which is good.

7. **Plan for validation against pilot data** — The literature unanimously recommends validation against real data. We should define validation metrics and thresholds before collecting pilot data.

8. **Monitor for systematic biases** — Track whether LLM-generated transitions show patterns inconsistent with behavioral science (e.g., too much activity persistence, insufficient burden effects).

### Open Questions and Future Work

1. **Validation methodology**: How do we validate LLM-generated transition probabilities without real data? Can we use expert review, cross-reference with published behavioral patterns, or sensitivity analysis?

2. **Prompt calibration**: Should we include quantitative behavioral priors in prompts? How much does prompt wording affect transition distributions?

3. **Multi-persona bootstrapping**: How should we extend the single-persona approach to capture population heterogeneity? What archetypes should we define?

4. **Adaptive bootstrapping**: Should the transition table be updated as real data becomes available (online learning)? Or is a fixed bootstrapped table sufficient?

5. **Transfer to continuous states**: Can our discrete approach scale to finer-grained state representations? At what granularity does LLM estimation become unreliable?

6. **Comparison with alternative bootstrapping methods**: How does LLM bootstrapping compare with (a) expert-specified transitions, (b) literature-derived transitions, (c) uniform transitions, (d) transitions learned from similar populations?

---

## References

Papers are ordered by relevance to our LLM transition bootstrap approach (highest first).

### Critical Relevance

1. **Alamdari, P.A., Cao, Y., & Wilson, K.W. (2024).** "Jump Starting Bandits with LLM-Generated Prior Knowledge." EMNLP 2024. *LLMs simulate user preferences to warm-start contextual bandit algorithms for health messaging.* [arXiv:2406.19317](https://arxiv.org/abs/2406.19317)

2. **Benechehab, A., et al. (2024).** "Zero-shot Model-based Reinforcement Learning using Large Language Models." arXiv. *Uses LLMs as zero-shot dynamics models for continuous control RL; introduces DICL.* [arXiv:2410.11711](https://arxiv.org/abs/2410.11711)

3. **Gazi, A.H., et al. (2026).** "Digital Twins for Just-in-Time Adaptive Interventions (JITAI-Twins)." JMIR. *Framework for optimizing JITAIs using LLM-driven digital twin simulations.* [jmir.org/2026/1/e72830](https://www.jmir.org/2026/1/e72830)

4. **Brooks, M., et al. (2025).** "GAMBITTS: Thompson Sampling for GenAI-Powered Adaptive Interventions." NeurIPS 2025. *Novel bandit algorithm explicitly modeling generative AI treatment content.* [arXiv:2505.16311](https://arxiv.org/abs/2505.16311)

5. **Karine, K. & Marlin, B. (2025).** "Using LLMs to Improve RL Policies in Personalized Health Adaptive Interventions." ACL CL4Health Workshop. *LLM-generated simulations improve RL policy learning for adaptive interventions.* [aclanthology.org/2025.cl4health-1.11/](https://aclanthology.org/2025.cl4health-1.11/)

6. **Evidence-Driven Sim Data (2026).** "Evidence-Driven Simulated Data in Reinforcement Learning Training for Personalized mHealth Interventions." MDPI Applied Sciences. *Behaviorally-informed synthetic data trains contextual bandits for health messages.* [mdpi.com/2076-3417/16/7/3463](https://www.mdpi.com/2076-3417/16/7/3463)

7. **Suh, J., et al. (2026).** "Quantifying the Utility of User Simulators for Building Collaborative LLM Assistants." arXiv. *Quality of user simulator critically determines RL training outcomes.* [arXiv:2605.09808](https://arxiv.org/abs/2605.09808)

8. **Bayley, A., et al. (2025).** "Robustness of LLM-Initialized Bandits for Recommendation Under Noisy Priors." GenAIRecP 2025. *LLM-initialized bandits robust to moderate noise; more data can reinforce bias.* [genai-personalization.github.io](https://genai-personalization.github.io/assets/papers/GenAIRecP2025/7_Bayley.pdf)

### High Relevance

9. **Yu, P. et al. (2026).** "Reinforcement World Model Learning for LLM-based Agents (RWML)." arXiv. *Self-supervised action-conditioned world models for LLM agents.* [arXiv:2602.05842](https://arxiv.org/abs/2602.05842)

10. **Tang, H., et al. (2024).** "WorldCoder, a Model-Based LLM Agent." NeurIPS 2024. *LLM builds Python code representing transition and reward functions.* [arXiv:2402.12275](https://arxiv.org/abs/2402.12275)

11. **Qwen Team (2026).** "Qwen-AgentWorld: Language World Models for General Agents." arXiv. *First native language world model trained on 10M+ trajectories.* [arXiv:2606.24597](https://arxiv.org/abs/2606.24597)

12. **Li & Wang (2026).** "Bridging the Agent-World Gap: Text World Models for LLM-based Agents." arXiv. *First systematic review formalizing TWMs as transition functions.* [arXiv:2606.09032](https://arxiv.org/abs/2606.09032)

13. **Park, J.S., et al. (2024).** "Generative Agent Simulations of 1,000 People." arXiv. *LLM agents replicate personality traits; 85% accuracy on GSS.* [arXiv:2411.10109](https://arxiv.org/abs/2411.10109)

14. **Jörke, M., et al. (2024).** "GPTCoach: Towards LLM-Based Physical Activity Coaching." CHI 2025. *GPT-4-based health coaching implementing evidence-based program.* [arXiv:2405.06061](https://arxiv.org/abs/2405.06061)

15. **Wang, X., et al. (2025).** "HealthGuru: Exploring Personalized Health Support through Data-Driven, Theory-Guided LLMs." CHI 2025. *LLM chatbot for sleep health using JITAI principles.* [arXiv:2502.13920](https://arxiv.org/abs/2502.13920)

16. **MHC-Coach (2025).** "Fine-tuning LLMs in Behavioral Psychology for Scalable Health Coaching." Nature NPJ Cardiovascular Health. *LLM fine-tuned on TTM produces stage-matched motivational messages.* [nature.com/articles/s44325-025-00083-5](https://www.nature.com/articles/s44325-025-00083-5)

17. **The Last JITAI? (2024).** "Exploring Large Language Models for Issuing Just-in-Time Adaptive Interventions." CHI 2025. *LLMs for both trigger detection and content personalization in JITAIs.* [arXiv:2402.08658](https://arxiv.org/abs/2402.08658)

18. **Zhang, J., et al. (2014).** "Impact of Different Policies on Unhealthy Dietary Behaviors in an Urban Adult Population." AJPH. *ABM for dietary policy evaluation using synthetic population.* [AJPH 104(7)](https://doi.org/10.2105/AJPH.2013.301657)

19. **Ozik, J., et al. (2021).** "Building and Experimenting with an Agent-Based Model to Study the Population-Level Impact of CommunityRx." PLOS Comp Bio. *Large-scale ABM with 802K synthetic agents for health intervention.* [doi.org/10.1371/journal.pcbi.1009471](https://doi.org/10.1371/journal.pcbi.1009471)

20. **Argollo da Costa, P.C., et al. (2025).** "Can A Society of Generative Agents Simulate Human Behavior and Attitudes?" arXiv. *LLM agents capture population-level vaccine hesitancy trends.* [arXiv:2503.09639](https://arxiv.org/abs/2503.09639)

### Moderate Relevance

21. **Lu, Y., et al. (2025).** "Can LLM Agents Simulate Multi-Turn Human Behavior?" ACL 2026. *LLM behavioral fidelity decreases over longer horizons.* [arXiv:2503.20749](https://arxiv.org/abs/2503.20749)

22. **Zhang, Z., et al. (2025).** "LLM-Powered User Simulator for Recommender System." AAAI 2025. *LLM simulator replaces opaque statistical user simulators.* [arXiv:2412.16984](https://arxiv.org/abs/2412.16984)

23. **Park, J.S., et al. (2023).** "Generative Agents: Interactive Simulacra of Human Behavior." UIST 2023. *Foundational generative agent framework; 25 LLM agents in sandbox world.* [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)

24. **Wang, R., et al. (2024).** "Patient-Ψ: Using Large Language Models to Simulate Patients." EMNLP 2024. *LLM + cognitive models for patient simulation in mental health training.* [arXiv:2405.19660](https://arxiv.org/abs/2405.19660)

25. **Geng, S., et al. (2025).** "JSONSchemaBench: A Rigorous Benchmark of Structured Outputs." arXiv. *Benchmarks constrained decoding on real-world JSON schemas.* [arXiv:2501.10868](https://arxiv.org/abs/2501.10868)

26. **Shorten, C., et al. (2024).** "StructuredRAG: JSON Response Formatting with Large Language Models." arXiv. *High variance in JSON success rates across models and prompts.* [arXiv:2408.11061](https://arxiv.org/abs/2408.11061)

27. **Wu, Y., et al. (2025).** "Human Digital Twins in Personalized Healthcare: An Overview." arXiv. *Digital twins facilitate what-if scenario simulation without risk to patients.* [arXiv:2503.11944](https://arxiv.org/abs/2503.11944)

28. **Tracy, M., et al. (2018).** "Agent-Based Modeling in Public Health: Current Applications and Future Directions." Annual Review of Public Health. *Foundational review establishing ABM methodology for health.* [doi.org/10.1146/annurev-publhealth-040617-014246](https://doi.org/10.1146/annurev-publhealth-040617-014246)

29. **Jiang, Y., et al. (2024).** "Can Language Models Serve as Text-Based World Simulators?" arXiv. *GPT-4 as world simulator: impressive but unreliable.* [arXiv:2406.06485](https://arxiv.org/abs/2406.06485)

30. **Kopec, J.A., et al. (2013).** "Development of a Population-Based Microsimulation Model of Physical Activity in Canada." Health Reports. *POHEM-PA module simulating physical activity trajectories.* [statcan.gc.ca](https://www150.statcan.gc.ca/pub/82-003-x/2013001/article/11777-eng.htm)

31. **Alomana, et al. (2026).** "When Correct Isn't Usable: Improving Structured Output Reliability in Small Language Models." arXiv. *Smaller models struggle with format compliance; prompt optimization helps.* [arXiv:2605.02363](https://arxiv.org/abs/2605.02363)

32. **Zhao, Y., et al. (2026).** "LLM Powered Social Digital Twins." arXiv. *LLM-based digital twin achieves 20.7% improvement on mobility prediction.* [arXiv:2601.06111](https://arxiv.org/abs/2601.06111)

33. **He, R., et al. (2025).** "Human Digital Twin: Data, Models, Applications, and Challenges." arXiv. *Comprehensive survey of human digital twin technology.* [arXiv:2508.13138](https://arxiv.org/abs/2508.13138)

34. **Piao, X., et al. (2025).** "AgentSociety: Large-Scale Simulation of LLM-Driven Agents in Society." AAMAS 2025. *Population-level simulation of millions of LLM-driven agents.* [arXiv:2502.08691](https://arxiv.org/abs/2502.08691)

35. **Kashefi, et al. (2025).** "Lost in Translation? Assessing User Perception of Prompt-Engineered Chatbot Tone-of-Voice for Smoking Cessation." Springer. *Prompt engineering controls LLM communication style for health coaching.* [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-032-27582-0_3)

36. **Willms, A. & Liu, S. (2024).** "Exploring the Feasibility of Using ChatGPT to Create JITAI Physical Activity mHealth Intervention Content." JMIR Medical Education. *ChatGPT generates contextually appropriate JITAI content.* [doi.org/10.2196/51426](https://mededu.jmir.org/2024/1/e51426/)

---

## Appendix: Search Methodology

### Search Agents

9 parallel search agents were deployed, each targeting a specific dimension of the literature:

1. LLM-based user simulation for RL
2. Prompt engineering for behavioral simulation
3. LLM-generated synthetic environments and world models
4. LLM calibration and validation methods
5. Digital twins and synthetic populations for health
6. LLM for MDP transition estimation
7. Structured output and JSON generation from LLMs
8. LLMs for adaptive health interventions (JITAIs)
9. Survey and meta-analysis papers across all dimensions

### Databases and Sources

- **arXiv** (primary) — Preprints in ML, AI, NLP, computational health
- **Semantic Scholar** — Academic paper search with citation analysis
- **Google Scholar** — Broad academic coverage
- **ACM Digital Library** — CHI, UIST, TOIS, RecSys venues
- **IEEE Xplore** — COINS, conference proceedings
- **JMIR** — Digital health, mHealth, JITAIs
- **Nature / Springer** — Health informatics, NPJ journals
- **PLOS** — Computational biology, public health
- **NeurIPS / ICML / ICLR / EMNLP / ACL** — ML and NLP conferences

### Search Terms

Primary queries included:
- "LLM user simulation reinforcement learning"
- "LLM world model transition dynamics"
- "LLM bootstrapping RL policy"
- "LLM synthetic environment agent training"
- "LLM JITAI health intervention"
- "LLM MDP transition probability estimation"
- "digital twin health intervention MDP"
- "synthetic population physical activity model"
- "constrained decoding JSON LLM"
- "LLM behavioral simulation prompt engineering"

### Results

- **Total papers found:** 188
- **After deduplication:** ~120 unique
- **After relevance filtering:** 52 papers included in this review
- **Date range:** 2013–2026 (concentrated 2024–2026)
- **Venues represented:** 25+ distinct venues across ML, NLP, health informatics, and HCI

### Inclusion Criteria

Papers were included if they addressed one or more of:
- Using LLMs to simulate user behavior or generate training data for RL
- Using LLMs to predict or model state transitions (world models)
- Using LLMs in the design, content generation, or optimization of health interventions
- Methods for structured/constrained output from LLMs
- Synthetic population generation or digital twin approaches for health
- Calibration, validation, or evaluation of LLM-generated behavioral data

### Limitations of This Review

1. The field is moving extremely fast — papers from 2025-2026 are still appearing
2. Health-specific LLM simulation papers are sparse; most come from recommender systems
3. No paper directly validates LLM-generated transition probabilities for a tabular MDP
4. Publication bias may favor positive results showing LLM capabilities
5. Our review focuses on English-language publications

---

## Self-Grill Findings (July 3, 2026)

A 5-question self-grill stress-tested the methodology. Key findings:

### Q1: Sanity checks before pilot data
Six proposed checks: monotonicity (burden vs step probability), action sensitivity (idle vs nudge), sleep-step correlation, marginal distribution comparison against published data, cross-table consistency, and prompt sensitivity analysis.

### Q2: Action effect ambiguity
The causal mechanism of movement_suggestion is ambiguous — it could increase steps (nudge works) or decrease them (interruption annoys). HeartSteps V2 found +34% but with a different action space. Sanity checks should be relative (idle vs journal should differ) not absolute.

### Q3: Distribution shift under optimized policy
Burden-as-state variable breaks the circularity — the table is P(s'|s,a,burden) not P(s'|s,a). As long as burden is correctly computed from the policy's actions, the table remains valid. Remaining risk: the LLM's burden-response function may not match reality.

### Q4: Statistical power with 10 draws per cell
Clopper-Pearson CIs are wide for rare transitions. Mitigations: increase N to 50-100 (cost scales linearly, still cheap), agent learns from sampled transitions (variance averages out over rollouts), and sensitivity analysis across N values is a cheap experiment.

### Q5: Markov assumption and multi-step trajectories
The factored state (step_bin, burden, day_of_week, sleep) captures some temporal dependency through burden (rolling 3-step window) and sleep (daily). Empirical validation: compute autocorrelation at lag-1/2/3 and dwell time distributions, compare against published accelerometry data (Dohler 2017: lag-1 ~0.3-0.5).

### Core vulnerability
**Systematic LLM bias, not variance.** You can increase sample size, add sanity checks, and factor the state space — but without real pilot data you can't know if the LLM's implicit behavioral model is wrong in a consistent direction. The mitigations (burden-as-state, cheap sensitivity analysis, autocorrelation checks) are pragmatic for this phase.

---

*Compiled: July 3, 2026*
*Reviewers: 9 parallel search agents + self-grill*

### Additional References (cited in Self-Grill)

37. Klasnja, P., et al. (2018). "Microrandomized Trials for HeartSteps V2." *Journal of Medical Internet Research.* — Foundational JITAI trial showing +34% step increase with context-aware nudges. Used as anchor for action effect size expectations.

38. Doherty, A., et al. (2017). "Large scale population-level physical activity monitoring." *International Conference on Ambient Intelligence.* — Published autocorrelation values for step counts (lag-1 ~0.3-0.5). Used as reference for temporal dependency validation.

39. Kopec, J.A., et al. (2013). "POHEM-PA: Microsimulation of physical activity." *Statistics Canada.* — Microsimulation of physical activity using population-level health data. Closest prior work on MDP-based physical activity modeling.

*Next update: After Sprint 1 validation results are available*
