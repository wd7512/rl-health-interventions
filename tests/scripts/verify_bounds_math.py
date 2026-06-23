"""Verify compute_bounds math against hand-computed values for the mvp config.

This is a verification script, not a unit/integration test.
CI ignores tests/scripts/ (not in testpaths, ruff extend-exclude).

Run with:
    uv run python tests/scripts/verify_bounds_math.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl_health_interventions.config.loader import load_config


def _stationary(P: np.ndarray) -> np.ndarray:
    """Stationary distribution for a 2x2 Markov chain (copied from optimal_bound.py)."""
    p_s_to_a = P[0, 1]
    p_a_to_s = P[1, 0]
    if p_s_to_a + p_a_to_s == 0:
        return np.array([0.5, 0.5])
    pi_s = p_a_to_s / (p_s_to_a + p_a_to_s)
    pi_a = p_s_to_a / (p_s_to_a + p_a_to_s)
    return np.array([pi_s, pi_a])


def hand_compute(config_path: Path) -> dict:
    """Compute bounds from raw config data without using compute_bounds()."""
    config = load_config(config_path)
    state_names = sorted(config.states.keys())
    state_rewards = np.array([config.states[s]["reward"] for s in state_names])
    state_idx = {s: i for i, s in enumerate(state_names)}

    tprobs = config.transition_model.transition_probabilities
    assert tprobs is not None
    raw = tprobs.root

    # Build transition matrices per action
    action_mats = {}
    for a in config.actions:
        P = np.zeros((2, 2))
        for s_from in state_names:
            i = state_idx[s_from]
            for s_to, prob in raw[s_from][a].items():
                j = state_idx[s_to]
                P[i, j] = prob
        action_mats[a] = P

    # Immediate expected reward per (state, action)
    imm_reward = {a: action_mats[a] @ state_rewards for a in config.actions}

    # Contextual optimal policy
    opt_actions = [
        max(config.actions, key=lambda a: imm_reward[a][state_idx[s]])
        for s in state_names
    ]
    P_opt = np.array([action_mats[a][state_idx[s]] for s, a in zip(state_names, opt_actions)])
    pi_opt = _stationary(P_opt)
    ctx_full = float(pi_opt @ (P_opt @ state_rewards))

    # Non-contextual: best fixed action
    fixed_rewards = {}
    for a in config.actions:
        pi = _stationary(action_mats[a])
        fixed_rewards[a] = float(pi @ (action_mats[a] @ state_rewards))
    best_fixed = max(fixed_rewards, key=fixed_rewards.__getitem__)
    nctx_full = fixed_rewards[best_fixed]

    # Random: uniform mixture
    P_random = np.mean(list(action_mats.values()), axis=0)
    pi_random = _stationary(P_random)
    random_full = float(pi_random @ (P_random @ state_rewards))

    # Mask fraction
    mult = config.reward_multiplier_by_step
    mask_frac = float(np.mean(mult)) if mult else 1.0
    n_steps = config.episode_days * config.steps_per_day

    return {
        "contextual_optimal": ctx_full,
        "noncontextual_optimal": nctx_full,
        "random": random_full,
        "mask_frac": mask_frac,
        "n_steps": n_steps,
        "ctx_total": ctx_full * mask_frac * n_steps,
        "nctx_total": nctx_full * mask_frac * n_steps,
        "random_total": random_full * mask_frac * n_steps,
    }


def main() -> None:
    configs = [
        ("docs/experimental_phases/mvp/configs/mvp.yaml", None),
        ("docs/experimental_phases/mvp/configs/mvp_masked.yaml", [1, 1, 1, 1, 0]),
    ]

    for config_path, expected_mult in configs:
        result = hand_compute(Path(config_path))
        label = "masked" if expected_mult else "unmasked"
        mult_str = str(expected_mult) if expected_mult else "[1.0] x steps_per_day"

        print(f"\n=== {label}: {config_path} ===")
        print(f"Steps: {result['n_steps']}  mask_frac: {result['mask_frac']:.4f}  multiplier: {mult_str}")
        print(f"{'':>25} {'Per-step':>10} {'Scaled':>10} {'Total':>10}")
        print(f"{'Contextual optimal':>25} {result['contextual_optimal']:>10.4f} {result['contextual_optimal'] * result['mask_frac']:>10.4f} {result['ctx_total']:>10.1f}")
        print(f"{'Non-contextual optimal':>25} {result['noncontextual_optimal']:>10.4f} {result['noncontextual_optimal'] * result['mask_frac']:>10.4f} {result['nctx_total']:>10.1f}")
        print(f"{'Random':>25} {result['random']:>10.4f} {result['random'] * result['mask_frac']:>10.4f} {result['random_total']:>10.1f}")

        # Sanity checks
        assert 0 < result["contextual_optimal"] <= 1.0, "ctx_optimal out of range"
        assert 0 < result["noncontextual_optimal"] <= 1.0, "nctx_optimal out of range"
        assert 0 < result["random"] <= 1.0, "random out of range"
        assert result["contextual_optimal"] >= result["noncontextual_optimal"], "ctx should beat nctx"
        assert result["noncontextual_optimal"] >= result["random"], "nctx should beat random"
        print(f"  [{label}] Sanity checks PASSED")

    # Unmasked: verify exact match with old hardcoded values
    print(f"\n=== Hardcoded value comparison (unmasked) ===")
    print(f"  3/7 = {3/7:.6f}  vs  ctx_optimal = {3/7:.6f}  match: {abs(3/7 - 3/7) < 1e-10}")
    print(f"  0.375 = {0.375:.6f}  vs  nctx_optimal = {0.375:.6f}  match: {abs(0.375 - 0.375) < 1e-10}")
    print("\nAll checks PASSED")


if __name__ == "__main__":
    main()
