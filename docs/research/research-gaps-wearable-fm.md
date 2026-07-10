# Research Gaps: Wearable Foundation Models × RL Health Interventions

> **Date:** 2026-07-10
> **Source:** Swarm research (10 agents) on SensorFM (Google, Jul 2026) and related wearable FM literature
> **Status:** Landscape review — identifies actionable gaps for this project

## Executive Summary

Google's SensorFM (Jul 2026) is the largest wearable foundation model to date: pre-trained on 1T+ minutes of sensor data from 5M people. It learns general-purpose representations of human physiology that transfer to 35 health prediction tasks. However, **no published work connects wearable FM embeddings to reinforcement learning for health interventions**. This document maps the landscape and identifies concrete research gaps this project could fill.

---

## 1. The Wearable FM Landscape (2024–2026)

### Google Lineage
| Model | Year | Scale | Key Innovation |
|-------|------|-------|----------------|
| LSM-1 | ICLR 2025 | 40M hrs, 165K users | Established scaling laws for wearable FMs |
| LSM-2 | NeurIPS 2025 | 40M hrs | Adaptive & Inherited Masking (AIM) — missingness-native pre-training |
| SensorLM | NeurIPS 2025 | 59.7M hrs, 103K users | Sensor-to-language alignment (CLIP-style) |
| **SensorFM** | arXiv Jul 2026 | **1T+ min, 5M users** | Largest scale, LLM-agentic head design, Personal Health Agent |

### Apple
| Model | Year | Scale | Key Innovation |
|-------|------|-------|----------------|
| PPG/ECG FM | ICLR 2024 | 141K users | Contrastive SSL on raw PPG/ECG |
| Accel FM | ICML 2025 | 172K users | Knowledge distillation from PPG to accelerometer |
| **WBM** | ICML 2025 | **2.5B hrs, 162K users** | Behavioral (not raw) sensor data, Mamba-2 backbone |

### Academic / Open-Source
| Model | Year | Scale | Key Innovation |
|-------|------|-------|----------------|
| **NormWear** | ACM TCH 2026 | 14.9K hrs (public) | Multi-modal, channel-aware attention, **open-source (Apache 2.0)** |
| NormWear 2.0 | arXiv 2026 | Same | Adds world modeling with latent dynamics |
| **Pulse-PPG** | UbiComp 2025 | 200M secs, 120 users | Field-trained PPG FM, **open-source (MIT)** |
| BioPM | ICML 2026 | Accelerometer | Movement-element transformer, **open-source** |
| **OpenMHC** | Stanford 2026 | Public benchmark | LSM-2 + WBM reimplementations, pretrained checkpoints on HuggingFace |
| JETS | NeurIPS workshop 2025 | 3M person-days | JEPA architecture, startup (Empirical Health) |
| PAT | IEEE JBHI 2026 | NHANES | Actigraphy-only FM for mental health |
| HiMAE | ICLR 2026 | ECG | Hierarchical MAE, 0.31M params |

---

## 2. Gap Analysis: FM × RL for Health Interventions

### Gap 1: No FM Embeddings Used as RL State [OPEN, PUBLISHABLE]

**Current state:** HeartSteps V2 (Liao et al., 2020) uses ~12 hand-crafted features as state for a contextual bandit: time of day, location, weather, prior step count, treatment burden. Every subsequent RL-for-JITAI paper has followed this pattern.

**What FMs offer:** SensorFM compresses 24h of multimodal physiology (PPG, accelerometer, EDA, temperature, SpO2) into a dense embedding. This encodes health states (sleep quality, stress, activity patterns) that hand-crafted features miss.

**Why nobody has done it:**
- Temporal granularity mismatch: FMs operate on daily windows; JITAIs need per-decision-point state (every 2–3 hours)
- No action-conditional dynamics: FMs don't model P(s'|s,a) — they're encoders, not world models
- Access: SensorFM, LSM, WBM are all research-only (no public weights)
- Safety: RL optimizing over black-box embeddings is hard to interpret

**What we could do:**
1. Use NormWear (open-source) or OpenMHC's LSM-2 checkpoint as embedding backbone
2. Start with FM embeddings as auxiliary features alongside hand-crafted state (not replacement)
3. Evaluate in StepCountJITAI simulation or our own environment
4. Measure: does adding FM embeddings improve policy performance over hand-crafted features alone?

**Difficulty:** Medium (engineering + simulation study, 6–12 months)
**Novelty:** High — no existing work

---

### Gap 2: Missing Data as Intervention Context [OPEN]

**Current state:** LSM-2/SensorFM's AIM treats all missingness equivalently — real gaps and artificial masks are modelled the same way.

**The gap:** Missingness patterns carry information. Device off = showering, swimming, or sleeping. Battery died = unexpected. In a JITAI context, knowing *why* data is missing changes the intervention decision.

**What we could do:**
1. Add missingness pattern features to our MDP state space
2. Model different missingness types (scheduled vs. unscheduled) as distinct state factors
3. Test whether missingness-aware state improves RL policy vs. naive handling

**Difficulty:** Low–Medium
**Novelty:** Medium — LSM-2 team acknowledges this but doesn't solve it

---

### Gap 3: Longitudinal Personalisation vs. Population Embeddings [OPEN]

**Current state:** All FMs produce fixed embeddings per time window. They learn population-level representations. HeartSteps adapts to individuals over months via RL policy updates.

**The gap:** FM embeddings don't capture individual trajectories — how THIS person's physiology changes over time. The "Beyond Static Encoders" position paper (Wu et al., 2026) argues this is the critical limitation of all current WFMs.

**What we could do:**
1. Fine-tune NormWear on user's own data over time (personalised adapter layers)
2. Maintain separate RL state that augments FM embeddings with user-specific features
3. Compare: population FM embedding vs. personalised embedding vs. hand-crafted features

**Difficulty:** Hard (fundamental research question)
**Novelty:** High — directly addresses the field's recognised gap

---

### Gap 4: FM as Reward Signal for RL [OPEN]

**Current state:** FM papers evaluate on classification/regression only. No FM has been used to generate reward signals for RL.

**Empirical Health (JETS)** explicitly proposes this: "deployed as a proxy reward in reinforcement learning (i.e. to tighten the feedback loop)." But they haven't implemented it.

**What we could do:**
1. Use SensorFM/NormWear predictions (e.g., predicted step count, sleep quality) as reward components
2. Compare FM-derived rewards vs. hand-crafted rewards vs. clinical labels
3. Test whether FM-based reward shaping improves sample efficiency

**Difficulty:** Medium
**Novelty:** High — Empirical Health proposes it but hasn't done it

---

### Gap 5: Multi-Timescale State Representation [OPEN]

**Current state:** FMs operate on daily windows. JITAIs need decisions at 2–5 hour intervals. HeartSteps uses 5 decision points per day.

**The gap:** Nobody has built a multi-timescale state representation that combines:
- Long-term FM embedding (daily health state)
- Short-term context (recent steps, location, weather)
- Intervention history (dosage, burden)

**What we could do:**
1. Design a hierarchical state: daily FM embedding + per-decision-point context features
2. Test different fusion strategies (concatenation, cross-attention, separate encoders)
3. Evaluate in our simulation environment

**Difficulty:** Medium
**Novelty:** Medium–High

---

### Gap 6: Causal Reasoning for Intervention Effects [HARD, LONG-TERM]

**Current state:** All FMs learn correlations, not causal models. RL needs counterfactual reasoning: "What would happen if I send a suggestion NOW?"

**What we could do:**
1. Use FM embeddings in a contextual bandit (simpler than full RL) to avoid causal requirements
2. Combine FM representations with causal inference methods (propensity scores, doubly-robust estimation)
3. This is HeartSteps V2's approach — FM embeddings as features in a doubly-robust estimator

**Difficulty:** Hard (active research area, 3–5+ years)
**Novelty:** Medium — HeartSteps already does this with hand-crafted features

---

## 3. Practically Usable Resources

| Resource | What | Access | URL |
|----------|------|--------|-----|
| **NormWear** | Pretrained FM + finetuning pipeline | Apache 2.0, HuggingFace | github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear |
| **OpenMHC** | Public benchmark, LSM-2/WBM reimplementations | Public, pip-installable | github.com/AshleyLab/OpenMHC |
| **Pulse-PPG** | PPG FM trained on field data | MIT, Zenodo weights | github.com/maxxu05/pulseppg |
| **BioPM** | Accelerometer FM | MIT, 3 checkpoints | github.com/Prithvitarale/biopm |
| **StepCountJITAI** | RL simulation environment for JITAIs | pip installable | Karine & Marlin, 2024 |
| **FLIRT** | Wearable data preprocessing | pip installable | PyPI: flirt |
| **OpenMHC leaderboard** | Comparison across models | HuggingFace Spaces | huggingface.co/spaces/MyHeartCounts/OpenMHC |

---

## 4. Recommended Priority Order

1. **Gap 1 (FM as RL state)** — Highest novelty, medium difficulty, directly relevant to our project. Start with NormWear embeddings + our simulation environment.

2. **Gap 4 (FM as reward signal)** — Empirical Health proposes this but hasn't done it. Could be a quick win if we have a working FM pipeline.

3. **Gap 5 (Multi-timescale state)** — Engineering challenge, but well-defined. Natural extension of Gap 1.

4. **Gap 2 (Missing data as context)** — Low-hanging fruit. Our environment already handles fragmented data; adding missingness features is straightforward.

5. **Gap 3 (Longitudinal personalisation)** — Important but hard. Requires user-level data over time.

6. **Gap 6 (Causal reasoning)** — Long-term, high-risk. Better to start with contextual bandits (Gap 1) than full causal RL.

---

## 5. How This Connects to Our Roadmap

| Roadmap Item | Relevant Gap |
|---|---|
| State space (2×4, physiological) | Gap 1, 5 — FM embeddings could inform state design |
| Reward (burden, non-activity) | Gap 4 — FM predictions as reward components |
| LLM transition bootstrapping | Gap 1 — FM embeddings as LLM input for transition modelling |
| HeartSteps V2 reproduction (PR #85) | Gap 1, 5 — test FM state against HeartSteps hand-crafted features |
| NHANES integration (#154) | Gap 1 — use OpenMHC benchmark for FM evaluation |

---

## References

- SensorFM: arXiv:2605.22759 (Jul 2026)
- LSM-2 / AIM: arXiv:2506.05321 (NeurIPS 2025)
- LSM-1: arXiv:2410.13638 (ICLR 2025)
- Apple WBM: arXiv:2507.00191 (ICML 2025)
- NormWear: arXiv:2412.09758 (ACM TCH 2026)
- "WFMs Beyond Static Encoders": arXiv:2603.19564 (Mar 2026)
- SensorLM: NeurIPS 2025
- OpenMHC: github.com/AshleyLab/OpenMHC
- Empirical Health JETS: NeurIPS 2025 TS4H workshop
- Personalized HeartSteps: arXiv:1909.03539 (2020)
- PH-LLM: Nature Medicine 2025
- Insulin Resistance Prediction: Nature 2026
