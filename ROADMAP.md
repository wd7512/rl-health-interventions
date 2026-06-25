# Roadmap

## Design decisions

 [x] [Initial Design](../docs/design/initial_design.pdf)
- [x] Sign off updated design direction ([[22.06 Chat with Swapnil and Mengyan]])
- [ ] Resolve D1-D14 with evidence (read papers first) — see [decision-catalogue](../docs/research/decision-catalogue.md)

## State space

- [x] MVP: binary (active/inactive)
- [ ] 2×4 (active/inactive × 4 personality types)
- [ ] State spaces informed with physiological data and more

## Action space

- [x] MVP: binary (nudge / no-nudge)
- [ ] 6-10 actions (including journalling, non-activity)

## Agents

- [x] Random (baseline)
- [x] Epsilon-Greedy
- [x] Decaying Epsilon-Greedy
- [x] UCB
- [x] Thompson Sampling
- [x] HeartSteps V2 (custom: Bayesian reward, dosage, proxy value function)
- [ ] DQN (Deep Q-Network)
- [ ] Policy Gradient / PPO

## Real data

- [ ] NHANES integration (data loaders exist, need evaluation)
- [ ] Synthetic-to-real transfer experiment

## LLM + high-dim

- [ ] LLM transition bootstrapping (simple)
- [ ] Cost-benefit analysis for more complex bootstrapping
- [ ] LLM transition bootstrapping (complex)

## Write-up

- [ ] Design doc + evidence
- [ ] Final report / paper draft
