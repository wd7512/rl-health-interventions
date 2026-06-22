"""Hyperparameter grid search for contextual bandit agents.

Grid:
  - epsilon_greedy:  epsilon [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
  - decaying_epsilon_greedy (D-EG):  epsilon_start [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                                     x decay_steps [25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700]
  - ucb:  c [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]

Each combination: 50 seeds, record mean total reward + last-50-step mean.
Runs both standard and contextual (context_feature="activity") variants.
Output: CSV to docs/mvp/hyperparam_results.csv
"""
from __future__ import annotations

import csv
import itertools
import logging
from pathlib import Path

import numpy as np

from rl_health_interventions.agents import derive_agent_seed, make as make_agent
from rl_health_interventions.config.loader import load_config
from rl_health_interventions.experiment import run_episode

logger = logging.getLogger(__name__)

EPSILON_VALUES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
DEC_EPSILON_START_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
C_VALUES = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
DECAY_STEPS_VALUES = [25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700]
N_SEEDS = 50

OUTPUT_PATH = Path(__file__).parent / "hyperparam_results.csv"
CTX_KWARGS = {"contextual": True, "context_feature": "activity"}


def _eg_grid(ctx: bool = False) -> list[tuple[dict, str]]:
    """Return (params, label) pairs for epsilon_greedy."""
    prefix = "ctx_" if ctx else ""
    suffix_kwargs = CTX_KWARGS if ctx else {}
    return [
        ({"epsilon": e, **suffix_kwargs}, f"{prefix}eps={e}")
        for e in EPSILON_VALUES
    ]


def _ucb_grid(ctx: bool = False) -> list[tuple[dict, str]]:
    """Return (params, label) pairs for ucb."""
    prefix = "ctx_" if ctx else ""
    suffix_kwargs = CTX_KWARGS if ctx else {}
    return [
        ({"c": c, **suffix_kwargs}, f"{prefix}c={c}")
        for c in C_VALUES
    ]


def _dec_grid(ctx: bool = False) -> list[tuple[dict, str]]:
    """Return (params, label) pairs for decaying_epsilon_greedy."""
    prefix = "ctx_" if ctx else ""
    suffix_kwargs = CTX_KWARGS if ctx else {}
    return [
        (
            {"epsilon_start": e, "epsilon_min": 0.01, "decay_steps": d, **suffix_kwargs},
            f"{prefix}eps_start={e},decay={d}",
        )
        for e, d in itertools.product(DEC_EPSILON_START_VALUES, DECAY_STEPS_VALUES)
    ]


# (agent_type, param_sweep) — each yields (params_dict, label_suffix)
AGENT_GRIDS = [
    ("epsilon_greedy", _eg_grid(ctx=False)),
    ("epsilon_greedy", _eg_grid(ctx=True)),
    ("decaying_epsilon_greedy", _dec_grid(ctx=False)),
    ("decaying_epsilon_greedy", _dec_grid(ctx=True)),
    ("ucb", _ucb_grid(ctx=False)),
    ("ucb", _ucb_grid(ctx=True)),
]


def run_one_config(
    config,
    agent_type: str,
    params: dict,
    n_seeds: int,
) -> tuple[float, float, float, float]:
    """
    Evaluates an agent configuration across multiple seeded episodes and computes aggregate reward statistics.
    
    Parameters:
    	config: Experiment configuration containing episode_days, steps_per_day, and actions.
    	agent_type: Type of agent to instantiate (e.g., "epsilon_greedy", "ucb").
    	params: Hyperparameter dictionary for the agent.
    	n_seeds: Number of independent episodes to run.
    
    Returns:
    	(total_mean, total_std, step_mean, last50_mean): A 4-tuple of floats representing:
    		- total_mean: Mean of total episode reward across all seeds.
    		- total_std: Standard deviation of total episode reward across seeds.
    		- step_mean: Mean reward per step, averaged across all steps and seeds.
    		- last50_mean: Mean reward over the final 50 steps across all seeds.
    """
    n_steps = config.episode_days * config.steps_per_day
    all_rewards = []
    for seed in range(1, n_seeds + 1):
        kwargs = dict(params)
        kwargs["actions"] = config.actions
        kwargs["seed"] = derive_agent_seed(seed, agent_index=0)
        agent = make_agent(agent_type, **kwargs)
        df = run_episode(config, agent, seed=seed)
        all_rewards.append(df["reward"].values)
    rewards = np.array(all_rewards)
    total_mean = float(rewards.sum(axis=1).mean())
    total_std = float(rewards.sum(axis=1).std())
    step_mean = float(rewards.mean())
    last50_mean = float(rewards[:, -50:].mean())
    return total_mean, total_std, step_mean, last50_mean


def main() -> None:
    """
    Run grid search over contextual bandit agent hyperparameters and write results to CSV.
    
    Loads the MDP configuration from config/rule_based.yaml, iterates over predefined
    agent types and parameter combinations, and executes each across multiple seeded
    episodes. Aggregates reward statistics (total reward mean/std, per-step mean, and
    mean of last 50 steps) and saves the results to CSV.
    """
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "docs" / "mvp" / "configs" / "mvp.yaml"
    config = load_config(str(config_path))
    n_steps = config.episode_days * config.steps_per_day

    logger.info("Config: %s", config_path)
    logger.info("MDP: %d days x %d steps = %d steps", config.episode_days, config.steps_per_day, n_steps)
    logger.info("Seeds per config: %d", N_SEEDS)

    rows = []
    for agent_type, grid in AGENT_GRIDS:
        for params, label in grid:
            logger.info("Running %s (%s)...", agent_type, label)
            total_mean, total_std, step_mean, last50_mean = run_one_config(
                config, agent_type, params, N_SEEDS
            )
            rows.append({
                "agent": agent_type,
                "params": label,
                "total_mean": total_mean,
                "total_std": total_std,
                "step_mean": step_mean,
                "last50_mean": last50_mean,
            })
            logger.info(
                "  total=%.1f ± %.1f  step=%.4f  last50=%.4f",
                total_mean, total_std, step_mean, last50_mean,
            )

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), OUTPUT_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
