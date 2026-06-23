---
title: "Literature Review — State & Action Space Design for RL Health Interventions"
status: "draft v0.1"
date: "2026-06-22"
purpose: "Synthesis — references to detailed research artifacts in action-space-design/ and state-space-design/"
related: "action-space-design/ · state-space-design/ · docs/initial_design.tex §3"
---

# Literature Review — State & Action Space Design for RL Health Interventions

> Executive synthesis. Detailed evidence and reference configurations are
> in `action-space-design/` and `state-space-design/`.

## Key findings

### Step count granularity

Dose-response meta-analyses (Saint-Maurice 2020, Paluch 2022) show mortality
risk plateaus at 6k–8k (older) / 8k–10k (younger). The 10k target is
arbitrary (1965 Japanese pedometer marketing). Use 4 threshold-based bins
(<4k, 4k–7k, 7k–10k, ≥10k) — not binary or continuous.

→ Deep dive: `state-space-design/step-bin-evidence.md`

### Action space

Current `feat/1a-actions` has 6 activity-only actions. Recommended set:
gentle_nudge, goal_nudge, journaling_prompt, sleep_hygiene,
social_encouragement, idle. Non-activity actions have strong standalone
RCT evidence but no published RL system has used them.

→ Deep dives:
  - `action-space-design/reference-configs.md` (6 systems)
  - `action-space-design/non-activity-interventions.md` (causal evidence)
  - `action-space-design/action-burden-evidence.md` (burden/fatigue models)

### State space

Recommended factored representation: step bin × time-of-day × day type ×
trend. Optional: stress, sleep (require EMA/wearable). Full factorial = 864
states → use contextual bandit conditioning, not tabular.

→ Deep dives:
  - `state-space-design/reference-configs.md` (6 systems)
  - `state-space-design/hidden-state-evidence.md` (mood/stress as latent state)
  - `state-space-design/step-bin-evidence.md`

### Reward

Proposed: `R(s,a,s') = base_reward(s') - burden_penalty(a) - fatigue_penalty`

**Unresolved**: Non-activity actions get zero step reward → agent avoids them.
Three options exist (see `action-space-design/non-activity-interventions.md`).
Fatigue model details are also unsettled (see `action-space-design/action-burden-evidence.md`).

### Open design decisions

The evidence does not settle these — flagged for resolution per implementation
branch:

1. Reward for non-activity actions (placeholder signal / defer / post-hoc)
2. Fatigue/burden model (separate or collapsed into hidden mood)
3. Burden penalty magnitudes (no empirical basis in literature)
4. Trend computation (window size, method)
5. Hidden mood state (no literature support for MVP inclusion)

See `action-space-design/README.md` and `state-space-design/README.md` for
detailed open questions.

## References

1. Saint-Maurice PF et al. (2020). *JAMA*, 323(12), 1151–1160.
2. Paluch AE et al. (2022). *Lancet Public Health*, 7(3), e219–e228.
3. Trella A et al. (2022). *Algorithms*, 15(8), 255.
4. Liao P et al. (2019). *ACM IMWUT*, 3(1), 1–22.
5. Klasnja P et al. (2019). *Annals of Behavioral Medicine*, 53(6), 546–557.
6. Smyth JM et al. (2018). *JMIR Mental Health*, 5(4), e11290.
7. Smyth JM et al. (1999). *JAMA*, 281(14), 1304–1309.
8. Norman SA et al. (2004). *Annals of Behavioral Medicine*, 27(2), 145–152.
9. Baikie KA, Wilhelm K (2005). *Advances in Psychiatric Treatment*, 11(5), 338–347.
10. Lee IM et al. (2022). *JAMA Internal Medicine*, 179(8), 1105–1112.
