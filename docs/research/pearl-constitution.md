# PEARL Constitution

**Pre-registered specification for validating LLM-generated synthetic patient data against the PEARL longitudinal study.**

- **Study:** PEARL (Physical activity Reinforcement Learning) — 4-arm RCT with n=7,711 in mITT analysis.
- **Purpose:** Verify that synthetic trajectories reproduce the distributions, effect sizes, and human behavioral patterns observed in real participants.
- **Date:** 2026-07-21
- **Status:** Pre-registered research spike.

---

## Tier 1 — Sanity Checks (must pass)

| ID | Name | Criterion | Rationale |
|----|------|-----------|-----------|
| T1.1 | Baseline stability | Mean steps during 30-day pre-study window within ±15% of PEARL reference (μ=5,618) | Ensure simulation starts from realistic baseline |
| T1.2 | Action differentiation | One-way ANOVA across 4 arms on mean daily steps at 1 month, p < 0.01 | Arms must diverge from each other |
| T1.3 | Direction correctness | RL arm mean daily steps > Control arm mean in ≥90% of seeds | RL must be at least non-inferior |
| T1.4 | No degenerate trajectories | No single-day step total == 0 or > 30,000 across any seed or arm | Physiological plausibility |

---

## Tier 2 — Distribution Checks

| ID | Name | Criterion | Rationale |
|----|------|-----------|-----------|
| T2.1 | Baseline mean | One-sample t-test vs μ=5,618 on 30-day pre-study window, p > 0.05 | Cannot reject PEARL baseline distribution |
| T2.2 | Effect size magnitude | RL vs Control Δ at 1 month: 150 ≤ Δ ≤ 450 steps | Effect must be within plausible range |
| T2.3 | Effect size ordering | Mean daily steps: RL > (Fixed ≈ Random ≈ Control) | PEARL observed RL superiority; other arms indistinguishable under random transitions |
| T2.4 | Attenuation pattern | 1-month to 2-month decay: 15% ≤ (Δ₁ - Δ₂)/Δ₁ ≤ 45% | Within PEARL-observed attenuation range (29%) |
| T2.5 | Between-person variance | ICC for daily steps across seeds: 0.4 ≤ ICC ≤ 0.9 | Participants should differ from each other |

---

## Tier 3 — Behavioural Realism Checks

| ID | Name | Criterion | Rationale |
|----|------|-----------|-----------|
| T3.1 | Burden saturation | Mean daily steps in RL arm peak then decline within 14–21 days of sustained nudging | Behavioural fatigue / habituation |
| T3.2 | Persona heterogeneity | One-way ANOVA across personas on effect size, p < 0.01, η² > 0.1 | Personas must produce meaningfully different responses |
| T3.3 | Weekend effect | Weekend daily steps < Weekday daily steps by 5–20% | Real-world activity pattern |
| T3.4 | Non-response detection | RL arm ≈ Control arm when action effects artificially set to zero | Mechanism check: no action → no differential effect |

---

## Tier 4 — Stress Tests (edge-case robustness)

| ID | Name | Criterion | Rationale |
|----|------|-----------|-----------|
| T4.1 | Random matrix | With randomised transition tables, all arms become indistinguishable: ANOVA p > 0.5 | Null model sanity check |
| T4.2 | Persona collapse | With identity transition (no state change), ANOVA across personas p > 0.5 | Null model sanity check |
| T4.3 | Infinite horizon | 365-day episodes: steps plateau (no unbounded growth or decay) | Long-run stability |
| T4.4 | Extreme demographics | 100% Resistant persona: RL vs Control Δ < 50 steps | Worst-case floor effect |

---

## Arm Mapping

| PEARL Arm | Simulation Agent | Description |
|-----------|-----------------|-------------|
| Control | FixedAgent(action="idle") | No nudges, pure observation |
| Random | RandomAgent (uniform) | Random action selection |
| Fixed | ComBWeightedFixedAgent | COM-B barrier-score weighted multinomial |
| RL | EpsilonGreedyAgent(epsilon=0.3) | ε-greedy contextual bandit approximation |

## Persona Weights (Demographic Matching)

| Persona | Weight | Rationale |
|---------|--------|-----------|
| base | 0.30 | Default population |
| goal_driven | 0.25 | Higher activity, goal-oriented |
| social_responder | 0.20 | Social influence sensitivity |
| stable_maintainer | 0.15 | Consistent but lower activity |
| resistant | 0.10 | Low engagement, floor effects |

Weights are tuned to approximate PEARL demographics: 86.3% female, mean age 42.1.

## Validation Protocol

1. Run all 4 tiers using `scripts/pearl_constitution/`
2. Record pass/fail for each check
3. Generate summary matrix
4. Flag any check that fails across ≥3 personas
5. Report to issue #246 for decision on next steps
