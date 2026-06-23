"""Derive theoretical optimal bounds from an MDP config.

Usage:
    uv run python docs/experimental_phases/mvp/optimal_bound.py configs/mvp.yaml
    uv run python docs/experimental_phases/mvp/optimal_bound.py configs/mvp_masked.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.config.schemas import MDPConfig
from _shared import resolve_config

logger = logging.getLogger(__name__)


def _stationary(P: np.ndarray) -> np.ndarray:
    """Stationary distribution π for a 2×2 Markov chain P (π·P = π)."""
    p_s_to_a = P[0, 1]
    p_a_to_s = P[1, 0]
    if p_s_to_a + p_a_to_s == 0:
        return np.array([0.5, 0.5])
    pi_s = p_a_to_s / (p_s_to_a + p_a_to_s)
    pi_a = p_s_to_a / (p_s_to_a + p_a_to_s)
    return np.array([pi_s, pi_a])


def compute_bounds(config: MDPConfig) -> dict:
    """Compute theoretical per-step expected reward bounds from an MDP config.

    Returns a dict with keys:
        contextual_optimal      — per-step E[r] under optimal context-aware policy
        noncontextual_optimal   — per-step E[r] under best fixed action
        random                  — per-step E[r] under uniformly random actions
        state_names             — sorted state names
        action_mats             — {action: transition_matrix} for each action
        imm_reward              — {action: immediate reward vector} for each action
        opt_actions             — optimal action per state
        ctx_stationary          — stationary distribution under optimal policy
        fixed_rewards           — {action: expected reward} for fixed-action policies
    """
    if not config.states:
        raise ValueError("config.states is required for computing bounds")
    if not config.actions:
        raise ValueError("config.actions is required for computing bounds")
    if not config.transition_model:
        raise ValueError("config.transition_model is required for computing bounds")
    state_names = sorted(config.states.keys())
    state_rewards = np.array([config.states[s]["reward"] for s in state_names])
    state_idx = {s: i for i, s in enumerate(state_names)}
    n_states = len(state_names)

    tprobs_obj = config.transition_model.transition_probabilities
    if tprobs_obj is None:
        raise ValueError("transition_probabilities is required for computing bounds")
    tprobs = tprobs_obj.root

    action_mats: dict[str, np.ndarray] = {}
    for a in config.actions:
        P = np.zeros((n_states, n_states))
        for s_from in state_names:
            i = state_idx[s_from]
            for s_to, prob in tprobs[s_from][a].items():
                j = state_idx[s_to]
                P[i, j] = prob
        action_mats[a] = P

    imm_reward: dict[str, np.ndarray] = {}
    for a in config.actions:
        imm_reward[a] = action_mats[a] @ state_rewards

    opt_actions: list[str] = []
    for s in state_names:
        i = state_idx[s]
        best_a = max(config.actions, key=lambda a: imm_reward[a][i])
        opt_actions.append(best_a)

    P_opt = np.array([action_mats[a][i] for i, a in enumerate(opt_actions)])
    ctx_stationary = _stationary(P_opt)
    ctx_full = float(ctx_stationary @ (P_opt @ state_rewards))

    fixed_rewards: dict[str, float] = {}
    for a in config.actions:
        π = _stationary(action_mats[a])
        fixed_rewards[a] = float(π @ (action_mats[a] @ state_rewards))
    best_fixed_action = max(fixed_rewards, key=fixed_rewards.__getitem__)
    nctx_full = fixed_rewards[best_fixed_action]

    P_random = np.mean(list(action_mats.values()), axis=0)
    π_random = _stationary(P_random)
    random_full = float(π_random @ (P_random @ state_rewards))

    return {
        "contextual_optimal": ctx_full,
        "noncontextual_optimal": nctx_full,
        "random": random_full,
        "state_names": state_names,
        "action_mats": action_mats,
        "imm_reward": imm_reward,
        "opt_actions": opt_actions,
        "ctx_stationary": ctx_stationary,
        "fixed_rewards": fixed_rewards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute theoretical bounds from a config"
    )
    parser.add_argument("config", type=str, help="Config filename in configs/")
    args = parser.parse_args()

    config_path = resolve_config(args.config)
    config = load_config(config_path)

    n_steps = config.episode_days * config.steps_per_day

    mult = config.reward_multiplier_by_step
    mask_frac = float(np.mean(mult)) if mult else 1.0

    try:
        bounds = compute_bounds(config)
    except ValueError as e:
        sys.exit(f"Error: {e}")
    ctx_full = bounds["contextual_optimal"]
    nctx_full = bounds["noncontextual_optimal"]
    random_full = bounds["random"]
    state_names = bounds["state_names"]
    action_mats = bounds["action_mats"]
    imm_reward = bounds["imm_reward"]
    opt_actions = bounds["opt_actions"]
    ctx_stationary = bounds["ctx_stationary"]
    fixed_rewards = bounds["fixed_rewards"]

    ctx_total = ctx_full * mask_frac * n_steps
    nctx_total = nctx_full * mask_frac * n_steps
    random_total = random_full * mask_frac * n_steps

    logger.info("Config: %s", config_path)
    logger.info(
        "Steps: %d (%d days x %d steps/day)",
        n_steps,
        config.episode_days,
        config.steps_per_day,
    )
    mult_str = str(mult) if mult else "[1.0] x steps_per_day"
    logger.info("Reward multiplier: %s  (mean=%.4f)", mult_str, mask_frac)
    logger.info("")
    logger.info("Optimal per-action (per full step):")
    for a in config.actions:
        π = _stationary(action_mats[a])
        logger.info(
            "  %s: E[r]=%.4f  (stationary π=%s, imm E[r]=%s)",
            a,
            fixed_rewards[a],
            np.round(π, 4).tolist(),
            np.round(imm_reward[a], 4).tolist(),
        )
    logger.info(
        "  random: E[r]=%.4f  (stationary π=%s)",
        random_full,
        np.round(_stationary(np.mean(list(action_mats.values()), axis=0)), 4).tolist(),
    )
    logger.info("")
    logger.info(
        "Contextual optimal policy: %s",
        ", ".join(f"{s}->{a}" for s, a in zip(state_names, opt_actions)),
    )
    logger.info("  stationary π=%s", np.round(ctx_stationary, 4).tolist())
    logger.info("")
    logger.info("%25s %10s %10s %10s", "", "Full step", "Scaled", "Total")
    logger.info(
        "%25s %10.4f %10.4f %10.1f",
        "Contextual optimal",
        ctx_full,
        ctx_full * mask_frac,
        ctx_total,
    )
    logger.info(
        "%25s %10.4f %10.4f %10.1f",
        "Non-contextual optimal",
        nctx_full,
        nctx_full * mask_frac,
        nctx_total,
    )
    logger.info(
        "%25s %10.4f %10.4f %10.1f",
        "Random",
        random_full,
        random_full * mask_frac,
        random_total,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
