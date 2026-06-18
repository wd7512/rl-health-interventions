# Issue 5: LLM bootstrapping for transition matrix generation (research spike)

**Labels:** `enhancement`, `research`
**Assignees:** @wd7512

## Context

Discussing next steps: *"Replacing the transition matrices with an LLM"*. Two approaches considered:

- **Option A (live):** Query an LLM at each timestep — 450 calls per episode, slow, expensive, nondeterministic
- **Option B (bootstrapping):** Query an LLM N≥100 times per `(s, a)` pair ahead of time, aggregate frequencies into a static transition matrix — one-time cost, deterministic, reusable

Mengyan's response: *"how'd you like to replace with LLM, using LLM to generate a static transition matrix?"* — she was interested in approach clarification, not a commitment to either.

## Scope

This is a **research spike** — a focused investigation with prototype code. The goal is to determine whether LLM-bootstrapped matrices are plausible enough to use as an alternative to hand-curated matrices.

## Specification

### Method

1. For each `(state, action)` pair, prompt an LLM describing the user's current state and the intervention taken:
   > "A sedentary user receives a motivational prompt to walk. What is their activity level at the next decision point? Choose: [active] or [sedentary]. Consider Reactance theory — reminders can help sedentary users but may overwhelm them."

2. Collect N responses for each `(s, a)` pair (N≥100).
3. Compute empirical frequency: `P(s' | s, a) = count(s') / N`
4. Produce a full transition probability matrix with:
   - Point estimates per entry
   - Bootstrap confidence intervals (95%)
   - Response consistency metrics (entropy of the distribution, majority vote confidence)

### Model options (evaluate at least 2)

| Model | Access | Cost | Notes |
|-------|--------|------|-------|
| Gemma 4 (local) | Local via Ollama | Free | Can run on laptop, evaluate speed/quality |
| GPT-4o mini / Claude Haiku | API | ~$0.01/query | Faster, higher quality, but costs for N×12 pairs |
| DeepSeek | API | Low | Strong reasoning, possible alternative |

### Evaluation criteria

1. **Plausibility:** Do the bootstrapped matrices reflect Reactance theory? (nudging helps sedentary ≥0.3, idle maintains ≥0.6)
2. **Consistency:** Low variance across independent runs with the same prompt
3. **Sensitivity:** Do different personas (specified in the prompt) produce different matrices?
4. **Comparison to hand-curated:** KL divergence between bootstrapped matrices and the hand-curated ones from Issue #3
5. **Downstream impact:** Run Thompson Sampling on a bootstrapped matrix vs hand-curated — do agent rankings change?

### Deliverables

1. **Prototype script** (`scripts/llm_bootstrap_transitions.py`):
   - Accepts model name, N, prompt template via CLI
   - Caches responses to disk (avoid repeat API costs)
   - Outputs matrix as YAML matching the `TransitionProbabilities` schema
   - Runs independently of the main experiment pipeline

2. **Evaluation report** (new `docs/sources/llm_bootstrap_evaluation.md`):
   - Matrices produced by each model
   - Bootstrap CIs per entry
   - KL divergence from hand-curated matrices (Issue #3)
   - Consistency metrics
   - Agent performance comparison (hand-curated vs LLM-bootstrapped)
   - Cost analysis: API tokens or time spent for local models

3. **Recommendation:**
   - Is LLM bootstrapping a viable alternative to hand-curation?
   - Which model gives the best quality/cost trade-off?
   - Should this be integrated into the config pipeline (as `transition_model: type: llm_bootstrapped`) or remain a research curiosity?

## Out of scope
- Live LLM integration in the experiment loop (Option A) — too slow for 450-step episodes
- Prompt engineering beyond a single baseline prompt per persona
- Production-grade integration into the config pipeline (depends on recommendation)

## Dependencies
- For API models: `openai` or `anthropic` SDK (optional dev dependency)
- For local models: Ollama (external tool, script interfaces via REST API)

## Related
- Issue #1 (extended action space — matrices need it)
- Issue #3 (hand-curated matrices — comparison target)
- `config/llm_based.yaml` (already exists as a placeholder)
- `config/learned.yaml` (placeholder for learned transitions)
