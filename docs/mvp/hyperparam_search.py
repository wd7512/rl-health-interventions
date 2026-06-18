"""Hyperparameter grid search for contextual bandit agents.

Grid:
  - epsilon_greedy:  epsilon [0.01, 0.05, 0.1, 0.2, 0.5]
  - decaying_epsilon_greedy:  epsilon_start [0.01, 0.05, 0.1, 0.2, 0.5]
                             x decay_steps [50, 100, 200]
  - ucb:  c [0.5, 1.0, 2.0, 3.0, 5.0]

Each combination: 50 seeds, record mean total reward + last-50-step mean.
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

EPSILON_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5]
C_VALUES = [0.5, 1.0, 2.0, 3.0, 5.0]
DECAY_STEPS_VALUES = [50, 100, 200]
N_SEEDS = 50

OUTPUT_PATH = Path(__file__).parent / "hyperparam_results.csv"

# (agent_type, param_sweep) — each yields (params_dict, label_suffix)
AGENT_GRIDS = [
    (
        "epsilon_greedy",
        [({"epsilon": e}, f"eps={e}") for e in EPSILON_VALUES],
    ),
    (
        "decaying_epsilon_greedy",
        [
            ({"epsilon_start": e, "epsilon_min": 0.01, "decay_steps": d}, f"eps_start={e},decay={d}")
            for e, d in itertools.product(EPSILON_VALUES, DECAY_STEPS_VALUES)
        ],
    ),
    (
        "ucb",
        [({"c": c}, f"c={c}") for c in C_VALUES],
    ),
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
    config_path = repo_root / "config" / "rule_based.yaml"
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
