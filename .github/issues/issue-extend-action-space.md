# Issue 1: Extend action space from 2 to 6 actions

**Labels:** `enhancement`, `phase-1`
**Assignees:** @wd7512

## Context

Design doc review with Mengyan Zhang (16 Jun 2026). The current MVP ships binary actions `{nudge, idle}` matching HeartSteps V2. Mengyan explicitly requested extending to the 6-action space defined in `initial_design.tex §3` as the top priority next exploration.

## Specification

### Current state
- Config schema requires a flat list of action names (e.g. `[nudge, idle]`)
- `RuleBasedTransition` builds a cache keyed by `(state, action)` — handles N actions already
- All 4 bandit agents internally assume binary actions in their update logic
- `MDPConfig` cross-reference validator checks transition probabilities cover exactly the declared actions

### Target action space (from `initial_design.tex §3`)

| Action | Label | Description |
|--------|-------|-------------|
| `a_0` | No message | Do not intervene |
| `a_1` | Motivational prompt | Generic encouragement |
| `a_2` | Walking suggestion | Recommend a short walk |
| `a_3` | Goal reminder | Remind of step goal |
| `a_4` | Recovery suggestion | Stretch or rest |
| `a_5` | Progress feedback | Report goal progress |

Each action needs two configurable penalties:
- `reward_penalty` — subtracted from the activity-change reward at the current epoch
- `burden_penalty` — added to the user's cumulative burden (fatigue model)

### Required changes

1. **Config schema** (`config/schemas.py`):
   - Add `ActionConfig` Pydantic model with fields: `name`, `reward_penalty` (default 0.0), `burden_penalty` (default 0.0)
   - Change `actions` field type from `list[str]` to `list[ActionConfig]` (backward-compatible with str parsing)
   - Update `MDPConfig` cross-reference validators to use action names extracted from configs

2. **Transition model** checks: `RuleBasedTransition` already handles arbitrary action keys via its dict cache. Verify with a 6-action test.

3. **Agents** (`agents/contextual_bandits/`):
   - All 4 agents use `self._actions` list — already variable-length
   - Thompson Sampling's Beta-Bernoulli update does not depend on action count
   - Epsilon-greedy and UCB use action-iterating Q-tables — already general
   - Verify no hardcoded binary assumptions remain in any agent

4. **Reward model** (`rewards/compound.py`):
   - Add `reward_penalty` subtraction: `r = base_reward[next_state] - action_config.reward_penalty`
   - Pass action config through to `CompoundReward`

5. **Config YAML**: Create `config/six_actions.yaml` with the full 6-action set and hand-curated placeholder transition probabilities

### New behaviour
- `CompoundReward.reward()` signature stays the same but subtracts action penalty
- Transition matrix dimensions go from 2×2×2 to 6×6×2 or general N×M×M
- All existing tests continue to pass with binary action configs (backward compatible)

## Deliverables

1. Updated `config/schemas.py` with `ActionConfig`
2. Updated `rewards/compound.py` with penalty subtraction
3. `config/six_actions.yaml` with full action set and placeholder transitions
4. Tests: verify reward penalty subtraction, verify agents handle 6 actions, verify backward compat with old configs
5. Run `uv run ruff check && uv run pytest` — all pass

## Out of scope
- Per-action burden accumulation in the environment (deferred to M-04 / Issue #3)
- Learned transition functions (Phase 2)

## Related
- `docs/design/initial_design.tex §3` (action space definition)
- `config/rule_based.yaml` (current 2-action config)
- Issue #2 (state space extension — complementary)
- Issue #3 (4 persona transition matrices — uses extended action set)
