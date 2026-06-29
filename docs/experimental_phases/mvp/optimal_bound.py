"""Derive theoretical optimal bounds from an MDP config."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.config.schemas import MDPConfig
from _shared import resolve_config

logger = logging.getLogger(__name__)


def _stationary(P: np.ndarray) -> np.ndarray:
    p_s_to_a = P[0, 1]
    p_a_to_s = P[1, 0]
    if p_s_to_a + p_a_to_s == 0:
        return np.array([0.5, 0.5])
    pi_s = p_a_to_s / (p_s_to_a + p_a_to_s)
    pi_a = p_s_to_a / (p_s_to_a + p_a_to_s)
    return np.array([pi_s, pi_a])


def _load_tprobs(config: MDPConfig) -> dict:
    table_dir_str = config.transition_model.table_dir
    if table_dir_str is None:
        raise ValueError("table_dir is required")
    table_dir = Path(table_dir_str)
    if not table_dir.exists():
        raise FileNotFoundError(f"Table directory not found: {table_dir}")
    tprobs: dict = {}
    for table_file in sorted(table_dir.glob("*.json")):
        data = json.loads(table_file.read_text())
        for state, actions in data.items():
            if state not in tprobs:
                tprobs[state] = {}
            for action, probs in actions.items():
                tprobs[state][action] = probs
    return tprobs


def compute_bounds(config: MDPConfig) -> dict:
    if config.state.schema is not None:
        raise ValueError(
            "config.state must contain actual state definitions "
            "(schema references are not supported)"
        )
    if config.state.factors is None:
        raise ValueError("state.factors is required")

    factor_configs = config.state.factors
    state_names = list(factor_configs[config.reward.factor].names)
    state_rewards_arr = np.array([config.reward.values[s] for s in state_names])
    state_idx = {s: i for i, s in enumerate(state_names)}
    n_states = len(state_names)

    if n_states != 2:
        raise ValueError(
            f"compute_bounds requires exactly 2 states (got {n_states}). "
            "_stationary() is specific to 2x2 transition matrices."
        )
    if not config.action_names:
        raise ValueError("config.actions is required")

    tprobs = _load_tprobs(config)
    if not tprobs:
        raise ValueError("No valid transition probabilities loaded")

    action_mats: dict[str, np.ndarray] = {}
    for a in config.action_names:
        P = np.zeros((n_states, n_states))
        for s_from in state_names:
            i = state_idx[s_from]
            row_ok = False
            for s_to in state_names:
                prob = tprobs.get(s_from, {}).get(a, {}).get(s_to, 0.0)
                j = state_idx[s_to]
                P[i, j] = prob
                if prob > 0:
                    row_ok = True
            if not row_ok:
                raise ValueError(
                    f"Missing transition row for state='{s_from}', action='{a}'"
                )
            rs = P[i].sum()
            if rs > 0:
                P[i] = P[i] / rs
        action_mats[a] = P

    action_penalties: dict[str, float] = {}
    for name, cfg in config.actions.items():
        penalty = (
            getattr(cfg, "action_penalty", 0.0)
            if hasattr(cfg, "action_penalty")
            else 0.0
        )
        action_penalties[name] = penalty
    pen_mult = config.reward.action_penalty_multiplier

    imm_reward: dict[str, np.ndarray] = {}
    for a in config.action_names:
        imm_reward[a] = action_mats[a] @ state_rewards_arr
        penalty = action_penalties.get(a, 0.0) * pen_mult
        imm_reward[a] = imm_reward[a] - penalty

    opt_actions: list[str] = []
    for s in state_names:
        i = state_idx[s]
        best_a = max(config.action_names, key=lambda a: imm_reward[a][i])
        opt_actions.append(best_a)

    P_opt = np.array([action_mats[a][i] for i, a in enumerate(opt_actions)])
    ctx_stationary = _stationary(P_opt)
    ctx_full = float(ctx_stationary @ (P_opt @ state_rewards_arr))

    fixed_rewards: dict[str, float] = {}
    for a in config.action_names:
        pi = _stationary(action_mats[a])
        fixed_rewards[a] = float(pi @ (action_mats[a] @ state_rewards_arr))
    best_fixed_action = max(fixed_rewards, key=fixed_rewards.__getitem__)
    nctx_full = fixed_rewards[best_fixed_action]

    P_random = np.mean(list(action_mats.values()), axis=0)
    pi_random = _stationary(P_random)
    random_full = float(pi_random @ (P_random @ state_rewards_arr))

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
    mult = config.reward.per_step_multiplier
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
    for a in config.action_names:
        pi = _stationary(action_mats[a])
        logger.info(
            "  %s: E[r]=%.4f  (stationary pi=%s, imm E[r]=%s)",
            a,
            fixed_rewards[a],
            np.round(pi, 4).tolist(),
            np.round(imm_reward[a], 4).tolist(),
        )
    logger.info(
        "  random: E[r]=%.4f  (stationary pi=%s)",
        random_full,
        np.round(_stationary(np.mean(list(action_mats.values()), axis=0)), 4).tolist(),
    )
    logger.info("")
    logger.info(
        "Contextual optimal policy: %s",
        ", ".join(f"{s}->{a}" for s, a in zip(state_names, opt_actions)),
    )
    logger.info("  stationary pi=%s", np.round(ctx_stationary, 4).tolist())
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
