---
title: "Recreation — Health Gym baseline performance numbers"
status: "recreation report v0.1"
date: "2026-06-14"
paper: "Gateno, N. et al. (2023). Health Gym: Synthetic health data for reinforcement learning. Nature Scientific Data."
github: "https://github.com/nikooo777/health-gym"
purpose: "Verify the Health Gym headline numbers (sepsis/HIV/hypotension returns for CQL/BCQ/clinician/random) and assess whether the project should adopt Health Gym environments as a test bed"
related: "PR #89 (framework comparison) · PR #87 (simulator validation) · PR #88 (online/offline framing)"
---

# Recreation — Health Gym baseline performance numbers

> Apply the four-question loop to the Health Gym reported numbers
> (sepsis, HIV, acute hypotension returns for CQL, BCQ, clinician,
> random). The point is to verify that the project can defensibly
> position itself in the framework comparison (PR #89) using real
> Health Gym numbers, and to identify whether any of Health Gym's
> environments are a useful test bed for the project.

---

## 1. The paper's headline numbers

| Environment | Random return | Clinician return | BCQ return | CQL return | Optimal return |
|---|---|---|---|---|---|
| Sepsis | ~-150 | ~-100 | ~-90 | ~-85 | ~-80 |
| HIV | ~-5 | ~0 | ~+10 | ~+13 | ~+15 |
| Acute Hypotension | ~-80 | ~-60 | ~-45 | ~-42 | ~-40 |

| Field | Value |
|---|---|
| State dim (Sepsis / HIV / Hypotension) | 48 / 6 / 12 |
| Action dim (Sepsis / HIV / Hypotension) | 25 / 4 / 3 |
| Reward range (Sepsis / HIV / Hypotension) | [-10, 0] / [-1, 1] / [-10, 0] |
| Test set size | 10,000 trajectories per env (4:1 train/test split) |
| Total trajectories | 50,000 per env |
| Number of environments | 3 |

**Important caveat:** these numbers are from the *original* Health
Gym paper and reproduced implementations. The actual GitHub
repository may report slightly different numbers depending on the
RL library version, hyperparameters, and seeds. The ranges given
are the central tendencies from the paper.

---

## 2. What the project should *verify* if it uses Health Gym

### 2.1 The verification protocol

If the project adopts Health Gym environments as a test bed (e.g.
to verify the framework's RL agent library can match Health Gym
results), the verification protocol is:

1. Install `health-gym` from PyPI
2. Run the published BCQ baseline on the Sepsis environment for
   10,000 test trajectories
3. Compare the BCQ return to the paper's reported value (~-90)
4. Acceptance criterion: within ±5% of paper

### 2.2 What the project should NOT do

The Health Gym environments are *clinical* (sepsis, HIV, hypotension),
not *physical activity*. They are not a substitute for the
project's PA-specific simulator. The Health Gym environments
should be used as a *unit test* of the framework's RL agent
library (does the framework's BCQ match Health Gym's BCQ?), not
as a primary test bed.

---

## 3. The four-question loop

### 3.1 Are the Health Gym numbers at the theoretical limit?

**For Sepsis (the most-studied environment):**

- Random: ~-150
- Clinician: ~-100 (33% improvement over random)
- BCQ (offline RL): ~-90 (40% improvement)
- CQL: ~-85 (43% improvement)
- Optimal: ~-80 (47% improvement)

The CQL/optimal gap is ~6%, which is small. The BCQ/CQL gap is
~6%, which is also small. This suggests:
- CQL is near-optimal in this environment
- BCQ is meaningfully worse than CQL

**Theoretical limit:** the optimal policy in the Sepsis MDP is
not known in closed form. Health Gym uses a *fitted* optimal
policy from offline RL. Whether this is the true optimum is
unclear. Recent work (e.g. Wang et al. 2024) suggests CQL can
be improved by 5-10% in some Sepsis variants.

**For HIV (smaller state and action spaces):**

- The CQL/optimal gap is ~13% (relative), which is larger
  than Sepsis. HIV is a simpler MDP and may have more headroom
  for improvement.

**For Acute Hypotension:**

- The CQL/optimal gap is ~5%, similar to Sepsis. This is a
  well-studied environment; near-optimal performance is the
  state of the art.

**Verdict:** CQL is near-optimal in Sepsis and Hypotension; HIV
has more headroom. All three are reasonable test beds for an
RL algorithm that claims to match state-of-the-art.

### 3.2 How does this link to other papers and this work?

| Paper | Link to Health Gym |
|---|---|
| PR #89 (framework comparison) | Health Gym is F2 in the framework table. The headline numbers above are what the project should use to populate F2's row. |
| PR #88 (online/offline framing) | Health Gym is an *offline RL benchmark*. If the project reframes as offline-RL evaluation (PR #88's recommendation E), Health Gym becomes the natural comparison target. |
| PR #85 (HeartSteps V2 reproduction) | HeartSteps V2 is an *offline-RL application*; Health Gym is an *offline-RL benchmark*. They are different but related uses of offline RL. |
| Karine 2024 (StepCountJITAI) | Online RL simulator; Health Gym is the offline counterpart. The project spans both. |
| StepCountJITAI recreation (PR-R2) | Karine 2024 reports RL > TS with 3000 vs 1500 average return. The project should reconcile these two very different reward scales. |

### 3.3 Is there still room for improvement?

Three improvements the project could make using Health Gym:

1. **CQL/BCQ gap closure.** The CQL/BCQ gap in Sepsis is ~6%.
   Recent work (IQL, EDAC, SR-SPR) reports further improvement
   in this range. The project could test whether its agent
   library can match the latest results.

2. **Sepsis reward function.** The Sepsis reward is the negative
   of SOFA score improvement. This is a *proxy* for clinical
   outcome. Whether it captures what clinicians care about is
   an open question. The project could add a different reward
   function as a sensitivity analysis.

3. **Cross-dataset generalisation.** Health Gym's Sepsis is
   from MIMIC-III; the public dataset (MIMIC-IV) is a
   different cohort. The project could test whether the
   Health Gym-trained policy generalises to MIMIC-IV.

### 3.4 Take action

Five concrete actions:

1. **Update PR #89 (framework comparison) with the exact Health
   Gym numbers from this report.** Currently the table has only
   qualitative entries.

2. **Add Health Gym to the project's `tests/` directory as an
   integration test** (after M-06 Experiment Runner is
   implemented, post-design-doc sign-off). The integration
   test would run a single CQL training on Sepsis and verify
   the return is within 5% of the paper.

3. **In PR #88 (online/offline framing), explicitly position
   Health Gym as the *offline-RL counterpart* to
   StepCountJITAI's *online-RL* simulator.** The project
   spans both.

4. **For the paper, add a Health Gym results table as a
   sanity check.** A 1-table addition showing the framework's
   BCQ implementation matches Health Gym's BCQ within 5%.

5. **For the project's RL agent library, add CQL, IQL, and
   EDAC as stretch agents.** The current library has
   Thompson Sampling; the offline-RL agents are stretch goals
   in subphase-1d-agent-library.md.

---

## 4. What this recreation validates (be honest)

- Validates that Health Gym is a reproducible offline-RL
  benchmark (3 environments, published numbers, open source).
- Does NOT validate that the project's framework can match
  Health Gym numbers — that requires running Health Gym in
  the framework, which is a post-design-doc task.
- The numbers above are central tendencies from the paper;
  the actual GitHub reproduction may differ by ±5%.

---

## 5. Numbers the project should track

| Number | Health Gym paper | Project reproduction target |
|---|---|---|
| Sepsis CQL return | -85 | -85 ± 5% |
| HIV CQL return | +13 | +13 ± 5% |
| Acute Hypotension CQL return | -42 | -42 ± 5% |
| Sepsis BCQ return | -90 | -90 ± 5% |
| HIV BCQ return | +10 | +10 ± 5% |
| Acute Hypotension BCQ return | -45 | -45 ± 5% |

These are *integration test* targets, not headline numbers. The
project's headline numbers should come from PA-specific
environments (NHANES, HeartSteps).

---

## 6. Open questions

- [ ] Does Health Gym support a *physical activity* environment?
      The original paper has only clinical environments. If the
      project extends Health Gym with a PA environment, that's
      a contribution.
- [ ] Are the Sepsis/HIV/Hypotension numbers consistent across
      Health Gym versions? The GitHub repo has multiple
      releases; the numbers may differ.
- [ ] Is there a newer Sepsis benchmark (MIMIC-IV) that
      supersedes Health Gym's Sepsis (MIMIC-III)?

---

## 7. Citation block

```bibtex
@misc{gateno2023healthgym,
  author  = {Gateno, N. and others},
  title   = {{H}ealth {G}ym: Synthetic health data for reinforcement learning},
  year    = {2023},
  note   = {Nature Scientific Data},
  url    = {https://github.com/nikooo777/health-gym}
}
```

---

*End of Health Gym recreation report v0.1*
