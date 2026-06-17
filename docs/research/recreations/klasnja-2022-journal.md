---
title: "Recreation — Klasnja 2022 npj Digital Medicine (HeartSteps V2 journal version)"
status: "recreation report v0.1"
date: "2026-06-14"
paper: "Klasnja, P. et al. (2022). A RL-enabled micro-randomized trial for physical activity promotion. npj Digital Medicine."
purpose: "Verify the journal-version headline numbers and assess how they differ from the Liao 2019 arXiv version"
related: "PR-R1 (Klasnja 2019 V1) · PR-R7 (Liao 2019 V2 arXiv) · PR #85 (V2 reproduction) · PR #90 (PR #85 review)"
---

# Recreation — Klasnja 2022 npj Digital Medicine

> Apply the four-question loop to the Klasnja 2022 paper, which
> is the journal version of the HeartSteps V2 study. The point
> is to verify the *published* headline numbers (which may
> differ from the 2019 arXiv version) and assess the gap
> between the journal version and the PR #85 reproduction.

---

## 1. The paper's headline numbers (target)

This is the citation in references.bib:

```bibtex
@article{klasnja2022heartsteps,
  author  = {Klasnja, P. and others},
  title   = {A {R}{L}-enabled micro-randomized trial for physical activity promotion},
  journal = {npj Digital Medicine},
  year    = {2022}
}
```

The DOI and page numbers are not in references.bib. This is a
gap that should be filled (see Tier 1 action 2 of the synthesis
report).

**Note on what we know vs. what we don't:** the journal version
is *not* in the survey, and we don't have the actual numbers.
This report is a *placeholder* — it documents what the project
should verify when the journal version is accessed, and what
the differences from the 2019 arXiv are likely to be.

---

## 2. What the journal version probably says

Based on the title ("RL-enabled micro-randomized trial") and the
context, the journal version likely reports:

1. **Final V2 study results** (not the pilot data the 2019
   arXiv was based on). The trial may have completed by 2022.
2. **Updated participant count** (potentially larger than 37,
   depending on the trial's actual enrolment).
3. **Updated effect-size numbers** (potentially different
   from the 78.4% / +29.75 pilot numbers in the 2019 arXiv).
4. **Methods refinements** (any improvements to the algorithm
   or evaluation since 2019).

Without access to the actual paper, these are educated guesses.
The project should fill this gap.

---

## 3. The four-question loop (with placeholders)

### 3.1 Is the journal version at the theoretical limit?

**Cannot be determined without the actual numbers.** The
project needs to:
- Access the published paper (via Bristol library or open
  access)
- Extract the headline numbers
- Compare to the 2019 arXiv version
- Compare to the PR #85 reproduction

### 3.2 How does this link to other papers and this work?

| Paper | Link to Klasnja 2022 |
|---|---|
| Liao 2019 (PR-R7) | The 2019 arXiv is the *pilot*; 2022 is the *final*. The PR #85 reproduction targets 2019. |
| Klasnja 2019 (PR-R1) | V1 is the *prior construction* data; V2 is the *RL algorithm*. 2022 is the V2 final. |
| PR #85 (HeartSteps V2 reproduction) | Targets the 2019 arXiv. If the 2022 numbers are different, the reproduction target should be updated. |
| PR #90 (PR #85 review) | Reviewer should re-check whether the 2019 or 2022 numbers are the right target. |

### 3.3 Is there still room for improvement?

Three open questions:

1. **Are the 2022 numbers consistent with the 2019 numbers?**
   Pilot data is often noisier than final data. The 2022
   numbers may have smaller CIs but similar point estimates.

2. **Are there methodological improvements in 2022 vs. 2019?**
   Possible: different RL algorithm, different prior
   construction, different evaluation metric.

3. **Is the 2022 trial *registered* and pre-registered?**
   If yes, this is a model for the project's own analysis
   plan (PR #86). If no, it's an opportunity for the
   project to improve on.

### 3.4 Take action

Five concrete actions:

1. **Fill in the Klasnja 2022 reference in references.bib.**
   Currently has year and journal but no DOI, volume, pages.
   *Effort: 10 minutes.*

2. **Access the published paper** (Bristol library or open
   access). Extract the headline numbers.
   *Effort: 30 minutes.*

3. **Compare 2022 numbers to 2019 arXiv.** Document the
   differences in a new section of PR #85's REPORT.md.
   *Effort: 30 minutes.*

4. **Update PR #85's reproduction target** if 2022 numbers
   differ from 2019. The reproduction should target the
   *published* version, not the arXiv.
   *Effort: 1 hour.*

5. **For the project's PR #86 (statistical analysis plan),
   cite Klasnja 2022 as a methodological precedent** for
   pre-registration of mHealth RL trials.
   *Effort: 30 minutes.*

---

## 4. What this recreation validates (be honest)

- This recreation is a *placeholder*. It does not validate
  any specific number.
- The validation will happen when the project accesses the
  journal version and updates PR #85's REPORT.md.
- The four-question loop is applied with the caveat that the
  numbers are not yet available.

---

## 5. Numbers the project should track (once accessed)

| Number | 2019 arXiv | 2022 journal | Project target |
|---|---|---|---|
| Participants | 37 (pilot) | TBD | TBD |
| % improved | 78.4% | TBD | TBD |
| Mean improvement | +29.75 | TBD | TBD |
| Episode length | 90 days | TBD | TBD |
| Re-runs | 96 | TBD | TBD |
| Methods refinements | baseline | TBD | TBD |

---

## 6. Open questions

- [ ] What is the Klasnja 2022 DOI? (Currently missing from
      references.bib.)
- [ ] Are the 2022 numbers consistent with the 2019 arXiv?
- [ ] Is the 2022 trial pre-registered? On ClinicalTrials.gov?
- [ ] Did the algorithm change between 2019 and 2022?
- [ ] Is there supplementary material with detailed tables?

---

## 7. Citation block

```bibtex
@article{klasnja2022heartsteps,
  author  = {Klasnja, P. and others},
  title   = {A {R}{L}-enabled micro-randomized trial for physical activity promotion},
  journal = {npj Digital Medicine},
  year    = {2022},
  note   = {DOI and pages to be added — see synthesis report Tier 1}
}
```

---

*End of Klasnja 2022 recreation report v0.1 (placeholder)*
