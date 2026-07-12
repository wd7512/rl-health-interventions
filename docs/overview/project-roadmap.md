---
title: "Roadmap"
status: "active"
last_reviewed: "2026-07-12"
---

# Roadmap

## Design decisions

- [x] [Initial Design](docs/decisions/initial_design.pdf)
- [x] Sign off updated design direction ([[22.06 Chat with Swapnil and Mengyan]])
- [ ] Resolve D1-D14 with evidence (read papers first) — see [decision-catalogue](docs/decisions/decision-catalogue.md)

## State space

- [x] MVP: binary (active/inactive)
- [ ] 2×4 (active/inactive × 4 personality types)
- [ ] State spaces informed with physiological data and more

## Action space

- [x] MVP: binary (nudge / no-nudge)
- [ ] 6-10 actions (including journalling, non-activity)

## Reward

- [x] MVP: step-count-based reward
- [ ] Burden model in reward (reduces responsiveness to actions)
- [ ] Non-activity reward (journalling, non-activity actions)
- [ ] Clinical data integration (3-week clinical results, synthetic)

## Agents

- [x] Random (baseline)
- [x] Epsilon-Greedy
- [x] Decaying Epsilon-Greedy
- [x] UCB
- [x] Thompson Sampling
- [ ] HeartSteps V2 (custom: Bayesian reward, dosage, proxy value function) — [PR #85](https://github.com/wd7512/rl-health-interventions/pull/85)
- [ ] DQN (Deep Q-Network)
- [ ] Policy Gradient / PPO

## Data pipeline

- [ ] NHANES integration — data loaders exist, need evaluation (#154)
- [ ] Synthetic-to-real transfer experiment
- [ ] Unify pandas/polars — remove pandas runtime dependency (#155)

## LLM + high-dim

- [ ] LLM transition bootstrapping (simple)
- [ ] Cost-benefit analysis for more complex bootstrapping
- [ ] LLM transition bootstrapping (complex)

## Write-up

- [ ] Design doc + evidence
- [ ] Final report / paper draft
