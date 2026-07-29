"""Analyse burden evolution across window sizes for the bayesian burden experiment.

Runs selected agents on each config, extracts burden at every step, and outputs:
  - Burden distribution per config (fraction of steps at low/medium/high)
  - Burden trajectory over time (binned by day)
  - Action distribution per config
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXPERIMENT_DIR))

from rl_health_interventions.config.loader import load_config  # noqa: E402
from rl_health_interventions.evaluation._shared import (  # noqa: E402
    run_agent_detailed,
)


# Configs to analyse
_BASELINE_CFG = (
    _REPO_ROOT
    / "docs/experimental_phases/sprint1_bootstrap/configs"
    / "sprint1_bootstrap_context_burden.yaml"
)

# Configs to analyse
_CONFIGS = {
    "baseline_det": _BASELINE_CFG,
    "bayes_w1d": _EXPERIMENT_DIR / "configs/sprint1_bootstrap_new_burden_w1d.yaml",
    "bayes_w3d": _EXPERIMENT_DIR / "configs/sprint1_bootstrap_new_burden_w3d.yaml",
    "bayes_w7d": _EXPERIMENT_DIR / "configs/sprint1_bootstrap_new_burden_w7d.yaml",
    "bayes_w7d_rebalanced": _EXPERIMENT_DIR
    / "configs/sprint1_bootstrap_new_burden_w7d_rebalanced.yaml",
    "bayes_w14d": _EXPERIMENT_DIR / "configs/sprint1_bootstrap_new_burden_w14d.yaml",
    "bayes_w30d": _EXPERIMENT_DIR / "configs/sprint1_bootstrap_new_burden_w30d.yaml",
}

# Agents to analyse
_AGENT_INDICES = {
    "Random": 0,
    "Std UCB": 7,
    "Std EG": 2,
    "Std D-EG": 4,
}

N_SEEDS = 50
BURDEN_MAP = {"low": 0, "medium": 1, "high": 2}


def analyse_burden() -> None:
    results: dict[str, dict] = {}

    for config_label, config_path in _CONFIGS.items():
        logger.info("\n=== %s ===", config_label)
        config = load_config(str(config_path))

        config_results: dict[str, dict] = {}
        for agent_name, agent_idx in _AGENT_INDICES.items():
            # Build agent config from the YAML's agents list
            agent_cfg = config.agents[agent_idx]
            logger.info("  Running %s...", agent_name)
            _, trajs = run_agent_detailed(
                config, agent_cfg, N_SEEDS, agent_index=agent_idx
            )

            # Extract burden values across all seeds
            n_steps = len(trajs[0])
            burden_matrix = np.zeros((N_SEEDS, n_steps), dtype=int)
            action_matrix = np.empty((N_SEEDS, n_steps), dtype="<U30")
            day_matrix = np.zeros((N_SEEDS, n_steps), dtype=int)

            for s, traj in enumerate(trajs):
                for t, rec in enumerate(traj):
                    burden_matrix[s, t] = BURDEN_MAP.get(rec.get("burden", "low"), 0)
                    action_matrix[s, t] = rec.get("action", "idle")
                    day_matrix[s, t] = rec.get("day", 0)

            # 1. Overall burden distribution
            total_steps = N_SEEDS * n_steps
            low_frac = float(np.sum(burden_matrix == 0) / total_steps)
            med_frac = float(np.sum(burden_matrix == 1) / total_steps)
            high_frac = float(np.sum(burden_matrix == 2) / total_steps)

            # 2. Burden by day (averaged across seeds and steps within day)
            max_day = int(day_matrix.max()) + 1
            daily_burden_mean = np.zeros(max_day)
            daily_low_frac = np.zeros(max_day)
            daily_med_frac = np.zeros(max_day)
            daily_high_frac = np.zeros(max_day)
            for d in range(max_day):
                mask = day_matrix == d
                if mask.any():
                    vals = burden_matrix[mask]
                    daily_burden_mean[d] = float(np.mean(vals))
                    n = len(vals)
                    daily_low_frac[d] = float(np.sum(vals == 0) / n)
                    daily_med_frac[d] = float(np.sum(vals == 1) / n)
                    daily_high_frac[d] = float(np.sum(vals == 2) / n)

            # 3. Action distribution
            all_actions = action_matrix.flatten()
            unique_actions, action_counts = np.unique(all_actions, return_counts=True)
            action_dist = {
                a: float(c / total_steps)
                for a, c in zip(unique_actions, action_counts, strict=True)
            }

            # 4. Idle fraction (burden impact on idling)
            idle_mask = all_actions == "idle"
            idle_frac = float(np.mean(idle_mask))

            config_results[agent_name] = {
                "burden_dist": {"low": low_frac, "medium": med_frac, "high": high_frac},
                "daily_burden_mean": daily_burden_mean.tolist(),
                "daily_low_frac": daily_low_frac.tolist(),
                "daily_med_frac": daily_med_frac.tolist(),
                "daily_high_frac": daily_high_frac.tolist(),
                "action_dist": action_dist,
                "idle_fraction": idle_frac,
                "n_steps": n_steps,
            }

            logger.info(
                "    burden: low=%.3f med=%.3f"
                " high=%.3f  idle=%.3f",
                low_frac,
                med_frac,
                high_frac,
                idle_frac,
            )

        results[config_label] = config_results

    # Write results
    out_path = _EXPERIMENT_DIR / "results" / "burden_evolution.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    analyse_burden()
