"""Derive theoretical optimal bounds from an MDP config.

Usage:
    uv run python docs/experimental_phases/mvp/optimal_bound.py configs/mvp.yaml
    uv run python docs/experimental_phases/mvp/optimal_bound.py configs/mvp_masked.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from rl_health_interventions.config.loader import load_config


def _stationary(P: np.ndarray) -> np.ndarray:
    """Stationary distribution π for a 2×2 Markov chain P (π·P = π)."""
    p_s_to_a = P[0, 1]
    p_a_to_s = P[1, 0]
    if p_s_to_a + p_a_to_s == 0:
        return np.array([0.5, 0.5])
    pi_s = p_a_to_s / (p_s_to_a + p_a_to_s)
    pi_a = p_s_to_a / (p_s_to_a + p_a_to_s)
    return np.array([pi_s, pi_a])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute theoretical bounds from a config")
    parser.add_argument("config", type=str, help="Config file path (relative to configs/ or absolute)")
    args = parser.parse_args()

    config_dir = Path(__file__).parent / "configs"
    raw = Path(args.config)
    if raw.is_absolute():
        config_path = raw
    else:
        config_path = config_dir / raw.name

    config = load_config(str(config_path))

    state_names = sorted(config.states.keys())
    state_rewards = np.array([config.states[s]["reward"] for s in state_names])
    state_idx = {s: i for i, s in enumerate(state_names)}
    n_states = len(state_names)
    n_actions = len(config.actions)
    n_steps = config.episode_days * config.steps_per_day

    mult = config.reward_multiplier_by_step
    mask_frac = float(np.mean(mult)) if mult else 1.0

    tprobs_obj = config.transition_model.transition_probabilities
    if tprobs_obj is None:
        sys.exit("Error: transition_probabilities is required for computing bounds")
    tprobs = tprobs_obj.root

    # Build P[a][s, s'] for each action
    action_mats: dict[str, np.ndarray] = {}
    for a in config.actions:
        P = np.zeros((n_states, n_states))
        for s_from in state_names:
            i = state_idx[s_from]
            for s_to, prob in tprobs[s_from][a].items():
                j = state_idx[s_to]
                P[i, j] = prob
        action_mats[a] = P

    # Immediate expected reward per (state, action)
    imm_reward: dict[str, np.ndarray] = {}
    for a in config.actions:
        imm_reward[a] = action_mats[a] @ state_rewards

    # Contextual optimal policy
    opt_actions: list[str] = []
    for s in state_names:
        i = state_idx[s]
        best_a = max(config.actions, key=lambda a: imm_reward[a][i])
        opt_actions.append(best_a)

    P_opt = np.array([action_mats[a][i] for i, a in enumerate(opt_actions)])
    ctx_stationary = _stationary(P_opt)
    ctx_full = ctx_stationary @ (P_opt @ state_rewards)

    # Non-contextual: best fixed action
    fixed_rewards: dict[str, float] = {}
    for a in config.actions:
        π = _stationary(action_mats[a])
        fixed_rewards[a] = π @ (action_mats[a] @ state_rewards)
    best_fixed_action = max(fixed_rewards, key=fixed_rewards.__getitem__)
    nctx_full = fixed_rewards[best_fixed_action]

    # Random: uniform action mixture
    P_random = np.mean(list(action_mats.values()), axis=0)
    π_random = _stationary(P_random)
    random_full = π_random @ (P_random @ state_rewards)

    ctx_total = ctx_full * mask_frac * n_steps
    nctx_total = nctx_full * mask_frac * n_steps
    random_total = random_full * mask_frac * n_steps

    print(f"Config: {config_path}")
    print(f"Steps: {n_steps} ({config.episode_days} days x {config.steps_per_day} steps/day)")
    mult_str = str(mult) if mult else "[1.0] x steps_per_day"
    print(f"Reward multiplier: {mult_str}  (mean={mask_frac:.4f})")
    print()
    print("Optimal per-action (per full step):")
    for a in config.actions:
        π = _stationary(action_mats[a])
        print(f"  {a}: E[r]={fixed_rewards[a]:.4f}  "
              f"(stationary π={np.round(π, 4).tolist()}, imm E[r]={np.round(imm_reward[a], 4).tolist()})")
    print(f"  random: E[r]={random_full:.4f}  "
          f"(stationary π={np.round(π_random, 4).tolist()})")
    print()
    print(f"Contextual optimal policy: {', '.join(f'{s}->{a}' for s, a in zip(state_names, opt_actions))}")
    print(f"  stationary π={np.round(ctx_stationary, 4).tolist()}")
    print()
    print(f"{'':>25} {'Full step':>10} {'Scaled':>10} {'Total':>10}")
    print(f"{'Contextual optimal':>25} {ctx_full:>10.4f} {ctx_full * mask_frac:>10.4f} {ctx_total:>10.1f}")
    print(f"{'Non-contextual optimal':>25} {nctx_full:>10.4f} {nctx_full * mask_frac:>10.4f} {nctx_total:>10.1f}")
    print(f"{'Random':>25} {random_full:>10.4f} {random_full * mask_frac:>10.4f} {random_total:>10.1f}")


if __name__ == "__main__":
    main()
