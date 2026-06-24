---
title: "Step Count Granularity — Dose-Response Evidence"
status: "draft v0.1"
date: "2026-06-23"
purpose: "Collate dose-response evidence supporting step count thresholds over continuous or binary encoding"
---

# Step Count Granularity — Dose-Response Evidence

> The evidence strongly favours coarse step categories (4 threshold-based bins)
> over fine-grained counts or a single binary threshold.

## 1. Key studies

| Study | N | Design | Outcome | Key finding |
|-------|---|--------|---------|-------------|
| Saint-Maurice (2020, JAMA) | 47,471 (15 cohorts) | Pooled meta-analysis | All-cause mortality | Risk drops steeply from ~2,700 to ~7,500, then **plateaus at 6k–8k** (older) / 8k–10k (younger) |
| Paluch (2022, Lancet Public Health) | 15 studies | Dose-response MA | All-cause and CV mortality | 1k-step increment → ~15% lower mortality; **flattens above ~8k** |
| Kraus (2019, Med Sci Sports Exerc) | ~20k (NHANES) | Cross-sectional | All-cause mortality | Biggest mortality jump: lowest quartile (<4k) → Q1 (~5.5k); **diminishing returns above ~7.4k** |
| Lee (2022, JAMA Intern Med) | 16,741 (women >60) | Prospective cohort | All-cause mortality | Minimal effective dose: **4k–4.5k** for 50% of optimal benefit |
| Ekelund (2019, BMJ) | 8 studies, 36,383 | Harmonised meta-analysis | All-cause mortality | **Diminishing returns above ~6k–8k**; sedentary time modifies the relationship |
| Tudor-Locke (2011, Int J Behav Nutr Phys Act) | Population norms | Review | Pedometer-determined steps | Population means: ~7k (older adults), ~10k (younger adults) |

## 2. The 10,000-step myth

The 10,000-step target traces to a **1965 Japanese pedometer marketing
campaign** ("manpo-kei" = 10,000 steps meter). No clinical trial has used
it as a primary endpoint. It persists because it is memorable, not
clinically optimal.

## 3. Clinical inflection points

From the dose-response curve, four clinically meaningful thresholds emerge:

| Threshold | Evidence | Interpretation |
|-----------|----------|----------------|
| < 4,000 steps | Below minimum effective dose (Lee 2022) | Highest mortality risk |
| 4,000–6,999 | Steep part of curve; biggest per-step benefit | High marginal return on intervention |
| 7,000–9,999 | Near plateau for older adults | Good — additional steps add little |
| ≥ 10,000 | Plateau region | Optimum — no additional mortality benefit |

## 4. Relevance for intervention (vs mortality)

All evidence above is for **mortality** outcomes. The dose-response curve
for *activity engagement* (probability of responding to a nudge) is unknown
and may differ. However:

- Step count categories are the only evidence-based threshold system
- The mortality plateaus provide *lower bound* justification — if the
  step-outcome curve plateaus at 8k for mortality, using finer granularity
  above 8k is almost certainly unnecessary for any clinically meaningful
  outcome
- If activity engagement declines at very high step counts (ceiling
  effects), a separate upper threshold may be needed — this is not
  addressed by mortality studies

## 5. Comparison to binary sedentary/active

| Metric | Binary (current) | 4 bins (proposed) |
|--------|-----------------|-------------------|
| Clinically meaningful distinction | Sedentary only vs everything else | 4 distinct regions with different benefit profiles |
| Agent information | Knows if user is moving or not | Knows how much user is moving relative to clinical thresholds |
| Risk of perverse policy | Agent may optimise for crossing a single threshold (5k) rather than sustained activity | Finer-grained signal reduces threshold-gaming risk |
| Implementation complexity | Trivial | Moderate (requires enumerating step count) |

## 6. Recommendation strength

The step bin thresholds are **the most evidence-supported recommendation**
in this review. Multiple large-scale meta-analyses converge on the same
inflection points. This is the one dimension where the literature is
essentially settled.
