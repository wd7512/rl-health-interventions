# Issue 4: Add DQN and policy gradient as full-RL baselines

**Labels:** `enhancement`, `phase-1`
**Assignees:** @wd7512

## Context

Mengyan asked: *"worth adding policy gradient/DQN as baselines?"*

All current agents are contextual bandits — they select actions independently at each timestep with no temporal credit assignment. The MVP spec documents a gap between the best bandit (~0.356 mean reward/step) and the optimal state-aware policy (~0.429), suggesting ~19% headroom. Full RL agents should close this gap by learning that nudging sedentary users and idling for active users is the optimal strategy.

## Current state

- 4 agents: Thompson Sampling, Epsilon-Greedy, UCB, Random
- All are stateless bandits (contextual mode conditions on current state only — no value bootstrapping)
- No experience replay, no target networks, no policy gradients
- `Agent` ABC has a simple `select_action(state) -> action` and `update(state, action, reward, next_state) -> None` interface

## Specification

### Required additions

1. **DQN agent** (`agents/dqn.py`):
   - Q-network: small MLP (state_dim → 64 → 64 → n_actions) using PyTorch
   - Experience replay buffer (configurable capacity, default 10,000)
   - Target network with soft/hard update (configurable `target_update_freq`)
   - Epsilon-greedy exploration with configurable decay schedule
   - Follows the existing `Agent` ABC interface
   - Register as `"dqn"` in the agent registry

2. **REINFORCE agent** (`agents/reinforce.py`):
   - Policy network: small MLP (state_dim → 64 → n_actions)
   - Monte Carlo return computation (full-episode rollouts)
   - Configurable learning rate and discount factor γ
   - Follows the `Agent` ABC interface
   - Register as `"reinforce"` in the agent registry

3. **Config validation** (`config/schemas.py`):
   - Add RL hyperparameter fields: `learning_rate`, `gamma`, `buffer_size`, `batch_size`, `target_update_freq`, `hidden_dim`
   - Validate these are present when `type: dqn` or `type: reinforce`
   - Reject RL hyperparameters for bandit agent types

4. **State encoding**:
   - Both RL agents need a numeric feature vector, not string state names
   - Add a `StateView.encode()` method that returns `np.ndarray` of floats
   - For the 2-state binary MVP: `[1.0 if activity == "active" else 0.0]`
   - For enriched state (Issue #2): include all numeric fields

### Behaviour

- DQN: state → Q-values → epsilon-greedy action → store (s, a, r, s', done) in buffer → sample batch → gradient step → periodic target update
- REINFORCE: collect full episode trajectory → compute discounted returns → gradient step on log-probabilities weighted by return

### Benchmark expectations

On `config/rule_based.yaml` (2 states, 2 actions, 450 steps):
- Random: ~0.301 (baseline lower bound)
- Thompson Sampling: ~0.356 (best current bandit)
- DQN target: ≥0.40 (should approach optimal ~0.429)
- REINFORCE target: ≥0.38 (simpler but higher variance)

## Deliverables

1. `src/rl_health_interventions/agents/dqn.py` — DQN agent
2. `src/rl_health_interventions/agents/reinforce.py` — REINFORCE agent
3. `src/rl_health_interventions/state.py` — add `StateView.encode()` method
4. Config schema updates for RL hyperparameters
5. `config/dqn_baseline.yaml` — DQN config for rule_based transitions
6. Tests: DQN learns on a simple deterministic environment, REINFORCE improves over random
7. Benchmark results in `docs/experiments/mvp/extensions.tex` (same format as MVP spec) comparing all 6 agents
8. `uv run ruff check && uv run pytest && uv run ty check` — all pass

## Out of scope
- PPO, SAC, or other advanced RL algorithms (Phase 2, if needed)
- CNN/LSTM encoders for raw sensor data (Phase 2)
- Multi-GPU or distributed training (Phase 2)

## Dependencies
- PyTorch (new dependency — add to `pyproject.toml`)
- Issue #2 (state space enrichment — richer state → more meaningful RL)

## Related
- `docs/experiments/mvp/mvp.tex §5` (bandit results showing the gap)
- `src/rl_health_interventions/agents/_base.py` (Agent ABC)
- `src/rl_health_interventions/config/schemas.py` (agent config validation)
