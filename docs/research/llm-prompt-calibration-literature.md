# LLM Magnitude Calibration & Prompt Reliability — Literature Review

> **Date:** 2026-08-01
> **Scope:** Evidence for the Option-B pipeline hardening (exemplar-magnitude leakage, rule placement, structured-output reliability, sampling variance) in the PEARL 12-action LLM bootstrapping pipeline.
> **Sources:** Crossref, PubMed, OpenAlex, arXiv — 28 verified records
> **Context:** `docs/research/prompt-refinement-log.md` rounds 1-10; full ladder analysis in the PEARL prompt pipeline (PR #286, #284).

---

## Executive Summary

We prompt an LLM to generate 7-day step-count histories per (state, action) cell and aggregate them into MDP transition tables. Ten refinement rounds surfaced three stubborn problems: **(P1)** magnitude calibration — the model overshoots/undershoots the +150-450 step target and exemplar totals leak into outputs as anchors; **(P2)** structured-output reliability — reasoning prefixes, ```json fences, truncation, duplicate days, concatenated objects; **(P3)** sampling variance — at 3 samples/cell, outlier cells (e.g. -614 lift) dominate.

**Short answer:** The three problems are all *documented phenomena* with mature evidence in adjacent fields, but **no published paper tests this exact pipeline** (LLM-bootstrapped MDP transition tables for step-count cells). The fixes below are graded by evidence strength.

## What the evidence supports (by problem)

### P1 — Magnitude calibration / exemplar anchoring

| Technique | Evidence | Strength |
|---|---|---|
| Remove absolute step totals from exemplars; express effects as deltas and ranges | Min et al. (#5, label-distribution leakage); Lou & Sun (#1, anchoring); Zhao et al. (#6) | **Strong** for the leakage mechanism; ranges-vs-points fix indirectly supported |
| Co-locate the ceiling constraint inside the operative rule paragraph, at prompt start/end, not middle | Liu et al. (#10, Lost in the Middle); Zhao et al. (#6, recency) | **Medium-strong** — position effects heavily replicated; matches our own r8-r10 observation |
| Keep hard constraints in the system role, exemplars as user role | Wallace et al. (#11, instruction hierarchy) | Medium (model-side mechanism) |
| Post-hoc numeric calibration via anchor-free probes | Zhao et al. (#6, contextual calibration concept) | Speculative |
| Temperature ~0.3 for magnitude-sensitive generation | Renze & Guven (#8) | Medium |

### P2 — Structured-output reliability

| Technique | Evidence | Strength |
|---|---|---|
| Grammar-constrained decoding (schema-aware token masking) | Geng et al. (#13) | **Strong** — guarantees syntax; not available via OpenRouter, so grammar *prompting* is our fallback |
| Grammar prompting (embed JSON schema in prompt) | Wang et al. (#14) | Medium — reduces but does not eliminate violations |
| Repeated sampling + validate-then-aggregate | Wang et al. (#9, self-consistency); Chen et al. (#22, USC) | **Strong** for sampling+aggregation; drop-and-aggregate alone is engineering practice |
| Multi-turn repair (feed failed JSON back) | none found | Speculative — last resort |

### P3 — Sampling variance / outliers / simulation validity

| Technique | Evidence | Strength |
|---|---|---|
| More samples/cell + robust estimator (median/trimmed/vote) instead of mean | Wang et al. (#9); Chen et al. (#22); Argyle et al. (#16); Park et al. (#18) | **Strong** as a general strategy; no tested n for step-count cells |
| Temperature ~0.3 (avoid 0 which collapses diversity; avoid high which inflates outliers) | Renze & Guven (#8) | Medium |
| Validate simulated step distributions against real reference data | Veenhuizen & O'Malley (#20); Santurkar et al. (#19); Aher et al. (#17); Dankar et al. (#21) | **Strong** — multiple studies show group-dependent distortion; validation is mandatory |
| Shrinkage / prior-pooling toward baseline for sparse cells | no direct citation; statistical principle | Speculative |

## What we would NOT trust

- **Prompt wording alone reliably enforcing numeric ceilings.** Instruction-based constraints are fragile (instruction hierarchy, lost-in-the-middle); constrained decoding enforces syntax, not magnitudes. Budget for post-hoc validation regardless of prompt design.
- **Small-sample cell means (n=3).** LLM numeric outputs are high-variance and outlier-prone; treat min-2-of-3 as a weak mitigation. Evidence points to substantially more samples and/or shrinkage.
- **"The model simulates real patients" without validation.** Grounded agents approach human test-retest reliability (82-86%, Park et al.) — the realistic ceiling — and carry systematic distortions (hyper-accuracy, demographic skew) that must be checked empirically.

## Annotated bibliography (28 records)

### P1 — Magnitude calibration, anchoring, numeric leakage

1. **Lou, J., & Sun, Y. (2025). Anchoring bias in large language models: an experimental study. *Journal of Computational Social Science*.** DOI: 10.1007/s42001-025-00435-2 (arXiv:2412.06593). GPT-4/Gemini show statistically reliable anchoring — an initial numeric value in the prompt shifts subsequent numeric judgments. Directly names the mechanism behind exemplar step totals acting as anchors.

2. **Macmillan-Scott, O., & Musolesi, M. (2024). (Ir)rationality and cognitive biases in large language models. *Royal Society Open Science*, 11(6).** DOI: 10.1098/rsos.240255. Seven LLMs display irrationality systematically different from humans' plus pronounced response inconsistency across runs — both the bias and the variance problem in one paper.

3. **Echterhoff, J., et al. (2024). Cognitive Bias in Decision-Making with LLMs.** arXiv:2403.00811. BiasBuster: 13,465-prompt dataset of prompt-induced/sequential/inherent biases (anchoring, framing); quantifies bias and tests self-debiasing prompts.

4. **Pagliaro, A. (2026). Cognitive Biases in Large Language Models: A Systematic Quantitative Assessment and Debiasing Analysis. *Electronics*, 15(11), 2428.** DOI: 10.3390/electronics15112428. Bias-Strength-Index across 8 LLMs / 11 bias categories: framing and primacy/recency near-universal; bias dominated by prompt-reformulation variance. (MDPI — corroboration only.)

5. **Min, S., Lyu, X., Holtzman, A., Artetxe, M., Lewis, M., Hajishirzi, H., & Zettlemoyer, L. (2022). Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? *EMNLP*.** DOI: 10.18653/v1/2022.emnlp-main.759. Randomly corrupting exemplar labels barely hurts classification; what drives outputs is the exemplars' **label space and label distribution**. The mechanistic explanation for "exemplar magnitudes leak into outputs" — the model copies the distribution of numbers it sees.

6. **Zhao, T. Z., Wallace, E., Feng, S., Klein, D., & Singh, S. (2021). Calibrate Before Use: Improving Few-Shot Performance of Language Models. *ICML*.** arXiv:2102.09690. Few-shot accuracy is unstable across prompt format/example choice/order; models are biased toward answers near the end of the prompt (recency). Proposes contextual calibration (estimate null-prompt bias, subtract it).

7. **Lu, Y., Bartolo, M., Moore, A., Riedel, S., & Stenetorp, P. (2022). Fantastically Ordered Prompts and Where to Find Them. *ACL*.** DOI: 10.18653/v1/2022.acl-long.556. Exemplar order alone swings few-shot performance from near-random to near-SOTA; no principled per-task ordering rule.

8. **Renze, M., & Guven, E. (2024). The Effect of Sampling Temperature on Problem Solving in Large Language Models. *Findings of EMNLP*.** DOI: 10.18653/v1/2024.findings-emnlp.432. Problem-solving accuracy peaks at low-to-moderate temperature (~0.3), degrades with both higher temperature (more variance/errors) and temperature 0 (repetition).

9. **Wang, X., Wei, J., Schuurmans, D., et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR*.** arXiv:2203.11171. Sampling multiple reasoning paths and taking the majority-consistent answer beats greedy decoding — the canonical "sample-many, aggregate-robustly" evidence.

### P2 — Instruction placement / binding and structured output

10. **Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024). Lost in the Middle: How Language Models Use Long Contexts. *TACL*, 12.** DOI: 10.1162/tacl_a_00638. In long contexts, models reliably use the beginning and end and systematically ignore the middle — the strongest evidence base for our observation that a standalone middle rule sentence doesn't bind while a constraint embedded in a salient paragraph does.

11. **Wallace, E., Xiao, K., Leike, R., Weng, L., Heidecke, J., & Beutel, A. (2024). The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions.** arXiv:2404.13208. Instruction-tuned models do not reliably prioritize system instructions over conflicting in-context text; proposes hierarchy training (model-side, not prompt-side). Explains why in-prompt exemplars can override stated rules.

12. **Lou, R., Zhang, K., & Yin, W. (2024). Large Language Model Instruction Following: A Survey of Progresses and Challenges. *Computational Linguistics*.** DOI: 10.1162/coli_a_00523. First comprehensive survey of instruction following; catalogs compliance factors (position, format, conflicting instructions, hierarchy).

13. **Geng, S., Josifoski, M., Peyrard, M., & West, R. (2023). Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning. *EMNLP*.** DOI: 10.18653/v1/2023.emnlp-main.674. Formal-grammar-constrained decoding (finite-state masking) guarantees output conformance to a target structure (e.g. JSON schema) without finetuning — the definitive technical fix for P2.

14. **Wang, B., Wang, Z., Wang, X., Cao, Y., Saurous, R. A., & Kim, Y. (2023). Grammar Prompting for Domain-Specific Language Generation with Large Language Models. *NeurIPS 36*.** DOI: 10.52202/075280-2837. Embedding an EBNF grammar in the prompt substantially reduces syntax errors when decoding-level control is unavailable (hosted APIs) — the vendor-agnostic fallback to #13.

15. **Shanahan, M., McDonell, K., & Reynolds, L. (2023). Role play with large language models. *Nature*, 623.** DOI: 10.1038/s41586-023-06647-8. Role-play prompting as emergent simulation behavior; persona/instruction framing governs behavior — the conceptual basis (and brittleness) of treating the model as a patient simulator.

### P3 — Sampling variance, outliers, and simulation validity

16. **Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). Out of One, Many: Using Language Models to Simulate Human Samples. *Political Analysis*, 31(3).** DOI: 10.1017/pan.2023.2. "Silicon samples" conditioned on real demographic backstories emulate response distributions of subpopulations — supports generating many synthetic patients per cell and requiring demographic conditioning.

17. **Aher, G., Arriaga, R. I., & Kalai, A. T. (2023). Using Large Language Models to Simulate Multiple Humans and Replicate Human Subject Studies. *ICML*.** arXiv:2208.10264. "Turing Experiments" replicate classic behavioral findings but find a "hyper-accuracy distortion" — simulated humans too accurate/optimal vs real humans.

18. **Park, J. S., Zou, C. Q., Kamphorst, J., et al. (2024, rev. 2026). LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals.** arXiv:2411.10109. Generative agents from 2-hour interviews + surveys on 1,052 Americans reach 82-86% of participants' own test-retest consistency — the strongest evidence that grounded LLM agents approximate individual-level behavior, and that the ceiling is human consistency itself.

19. **Santurkar, S., Durmus, E., Ladhak, F., Lee, C., Liang, P., & Hashimoto, T. (2023). Whose Opinions Do Language Models Reflect? *ICML*.** arXiv:2303.17548. LLM opinion distributions systematically misalign with US demographic groups even under demographic steering; misalignment varies strongly by group.

20. **Veenhuizen, M., & O'Malley, A. (2025). Demographic biases in AI-generated simulated patient cohorts: a comparative analysis against census benchmarks. *Advances in Simulation*, 10.** DOI: 10.1186/s41077-025-00385-9; PMID: 41254805. LLM-generated simulated patient cohorts deviate from census benchmarks on demographics — the closest health-domain analogue to our pipeline; simulated cohorts must be validated against real reference distributions.

21. **Huet-Dastarac, M., Dankar, F. K., Liu, D., et al. (2026). An Evaluation of Pretrained Generative Models for Augmenting Small Health Data: Comparative Modeling Study. *J Med Internet Res*.** DOI: 10.2196/88678; PMID: 42296511. Benchmarks generative models for augmenting small health datasets; large quality differences; poor-quality synthetic augmentation can hurt downstream models.

22. **Chen, X., Aksitov, R., Alon, U., et al. (2023). Universal Self-Consistency for Large Language Model Generation.** arXiv:2311.17311. Extends self-consistency to free-form outputs by having the LLM select the most consistent answer among candidates — the aggregation mechanism for step-count vectors where exact-match voting is impossible.

### Cross-cutting context: LLMs as world models / behavior simulators

23. **Andreas, J. (2022). Language Models as Agent Models. *Findings of EMNLP*.** DOI: 10.18653/v1/2022.findings-emnlp.423. Theoretical argument that LLMs trained on human text are, in a narrow sense, models of the agents that produced that text.

24. **Hao, S., Gu, Y., Ma, H., Hong, J., Wang, Z., Wang, D., & Hu, Z. (2023). Reasoning with Language Model is Planning with World Model. *EMNLP*.** DOI: 10.18653/v1/2023.emnlp-main.507. RAP: LLM as world model (state-transition function) with MCTS over rollouts — LLM-as-transition-model is viable in principle (closest analogue to our bootstrap).

25. **Li, K., Hopkins, A., Bau, D., Viégas, F., Pfister, H., & Wattenberg, M. (2023). Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task. *ICLR*.** arXiv:2210.13382. A GPT trained on Othello moves develops an internal board-state representation — LMs can encode latent dynamics beyond surface statistics.

26. **Yildirim, I., & Paul, L. A. (2024). From task structures to world models: what do LLMs know? *Trends in Cognitive Sciences*, 28(5).** DOI: 10.1016/j.tics.2024.02.008. Cautionary: LLM task performance often reflects in-context task structures rather than genuine transferable world models — treat LLM-generated dynamics as conditional statistical priors, not ground truth.

27. **Vezhnevets, A., et al. (2023). Generative Agent-Based Modeling ... using Concordia.** arXiv:2312.03664. Framework/library for LLM-driven agent-based social simulation.

28. **Bail, C. A. (2024). Can Generative AI improve social science? *PNAS*, 121(21).** DOI: 10.1073/pnas.2314021121. Balanced assessment of GenAI for simulating human behavior: opportunities plus documented limitations (training-data bias, homogeneity, validity).

---

## Application to our pipeline (Option B decisions)

| Pipeline change | Primary evidence | Status |
|---|---|---|
| Exemplar deltas/ranges, no absolute totals | #5, #1, #6 | Round 11 experiment |
| JSON schema embedded in system prompt | #14 | Round 11 experiment |
| Median/trimmed cell lift alongside mean | #9, #22 | Analyzer addition |
| Retry-on-parse-None (bounded) | engineering practice; #13 framing | Generator addition |
| Temperature 0.3 for full-scale run | #8 | Full-run decision (documented break with rounds 1-10) |
| Samples 5/cell at full scale | #9, #16, #18 | Full-run decision |
