# Phase 5, Domain 6: Research Integrity & Ethics Audit

**Date:** 2026-06-11
**Auditor:** Primary Model (Qwen 3.7 Max)

---

## Citation Completeness

**Claims in initial_design.tex that reference prior work:**

| Claim | Citation | Status |
|-------|----------|--------|
| HeartSteps V1 MRT feasibility | klasnja2019heartsteps | ✓ Cited |
| HeartSteps V2 RL-in-the-loop | klasnja2022heartsteps | ✓ Cited |
| StepCountJITAI simulation environment | karine2024stepcountjitai | ✓ Cited |
| Health Gym synthetic environments | gateno2023healthgym | ✓ Cited |
| ExtraSensory benchmark dataset | vaizman2017extrasensory | ✓ Cited |
| WISDM benchmark dataset | kwapisz2011wisdm | ✓ Cited |
| All of Us Fitbit Dataset | patten2026allofus | ✓ Cited |
| UK Biobank Accelerometer | williamson2024ukbiobank | ✓ Cited |

**Uncited claims:**
- "RL offers a principled approach to personalising JITAIs" — general claim, no specific citation needed
- "Existing simulators are bespoke" — implicit reference to StepCountJITAI and Health Gym, could be more explicit

**Score: 8/8 specific claims cited (100%)**

---

## Ethical Approval

**Status:** ✗ Not documented.

- No IRB/ethics approval documentation
- No mention of ethics review for health intervention research
- No discussion of ethical considerations for simulated health interventions

**Note:** Phase 1 uses synthetic data only, which may not require IRB approval. However, Phase 2 targets real health datasets (HeartSteps, All of Us, UK Biobank) which require ethics review.

---

## Data Consent and Privacy

**Status:** ✗ Not documented.

- No informed consent documentation
- No data anonymisation procedures
- No PHI/PII handling guidelines
- No GDPR/HIPAA compliance discussion
- No data use agreement documentation for Phase 2 datasets

**Phase 2 dataset access:**
- All of Us: Requires data use agreement, IRB approval
- UK Biobank: Requires material transfer agreement
- HeartSteps: Publicly available but originally collected under IRB protocol

---

## Bias and Fairness

**Status:** ✗ Not documented.

- No discussion of health equity considerations
- No demographic bias analysis in RL policy
- No disparate impact assessment
- No discussion of representation in synthetic data generation
- No fairness metrics defined

**Risk:** The 4 user archetypes (goal-driven, social, resistant, stable) may not capture demographic variation. If archetypes correlate with age, gender, or ethnicity, the RL policy could have disparate impact.

---

## RL Safety

**Status:** ✗ Not documented.

- No safety constraints for health interventions
- No off-switch mechanisms
- No human-in-the-loop requirements
- No discussion of conservative policy approaches
- Burden model is a soft constraint, not a hard safety guarantee

**Risk:** RL agent could recommend excessive interventions causing notification fatigue or adverse effects. No mechanism to prevent this.

---

## Reporting Guidelines Compliance

### ARRIVE (Animal Research)
**Not applicable** — no animal research.

### CONSORT (Clinical Trials)
**Status:** ✗ Not applicable at design stage, but should be planned.

The framework will eventually support simulated clinical trials. CONSORT guidelines should be considered for:
- Participant flow diagram (simulated users)
- Randomisation (action selection)
- Blinding (not applicable to simulation)
- Outcome measures (reward, adherence, behaviour change)

### SPIRIT (Protocols)
**Status:** ⚠️ Partial.

initial_design.tex serves as a protocol document but lacks:
- Pre-specified primary/secondary endpoints
- Sample size justification
- Statistical analysis plan
- Interim analysis rules

---

## Nature Research Integrity Checklist

| Item | Status | Notes |
|------|--------|-------|
| Methods detailed enough for reproduction | ⚠️ Partial | Design doc detailed, but no working code |
| Data availability statement | ⚠️ Partial | Synthetic data planned, Phase 2 datasets listed |
| Code availability | ⚠️ Partial | Repo exists but no release/DOI |
| Ethical approval | ❌ Missing | No IRB/ethics documentation |
| Informed consent | ❌ Missing | No consent documentation |
| Data privacy protection | ❌ Missing | No GDPR/HIPAA discussion |
| Bias assessment | ❌ Missing | No fairness analysis |
| Statistical methods pre-specified | ❌ Missing | No analysis plan |
| Sample size justification | ❌ Missing | No power analysis |
| Effect sizes reported | ❌ Missing | No results yet |
| Confidence intervals | ❌ Missing | No statistical analysis |
| Multiple comparison correction | ❌ Missing | No analysis plan |
| Limitations discussed | ⚠️ Partial | Decision Log has open questions |
| Competing interests | ❌ Missing | Not applicable yet |
| Author contributions | ❌ Missing | Not applicable yet |

**Completion: 2/15 items (13%)**

---

## Summary

**Citation completeness:** 100% (8/8 claims cited)
**Nature checklist:** 13% (2/15 items)
**Critical gaps:** No ethics documentation, no privacy compliance, no safety constraints, no statistical analysis plan, no bias assessment

**Top 3 Issues:**
1. No ethical approval or IRB documentation for Phase 2 real data use
2. No privacy/GDPR/HIPAA compliance documentation
3. No RL safety constraints for health interventions

---

*End of Domain 6 audit*
