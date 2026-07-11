"""Evaluate reference policy bounds for sprint1_bootstrap via simulation.

Compares fixed-action, random, and greedy-oracle policies by running
short seeded episodes through run_episode and reporting per-step reward.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from _shared import resolve_config

from rl_health_interventions.agents import derive_agent_seed
from rl_health_interventions.agents import make as make_agent
from rl_health_interventions.config.loader import load_config
from rl_health_interventions.episode import run_episode

logger = logging.getLogger(__name__)

N_SEEDS = 10

ALPHA = 0.9
STEP_BIN_VALUES = {"inactive": 0.0, "moderate": 0.5, "active": 1.0}
SLEEP_VALUES = {"good": 1.0, "poor": -1.0}
ACTION_PENALTIES = {
    "idle": 0.0,
    "movement_suggestion": 0.05,
    "goal_reminder": 0.05,
    "journal": 0.05,
}


class _FixedActionAgent:
    """Always returns the same action."""

    def __init__(self, action: str):
        self._action = action

    def select_action(self, _state) -> str:
        return self._action

    def update(self, state, action, reward, next_state) -> None:
        pass


class _GreedyOracleAgent:
    """One-step lookahead: picks action maximizing expected immediate reward.

    Uses the bootstrap within_day tables to evaluate next step_bin distribution
    for each action and computes E[reward | state, action].
    """

    def __init__(self, within_day_tables: list[dict], actions: list[str]):
        self._within_day = within_day_tables
        self._actions = actions

    def select_action(self, state) -> str:
        best_action = self._actions[0]
        best_value = -float("inf")
        for a in self._actions:
            value = self._expected_immediate_reward(state, a)
            if value > best_value:
                best_value = value
                best_action = a
        return best_action

    def update(self, state, action, reward, next_state) -> None:
        pass

    def _expected_immediate_reward(self, state, action: str) -> float:
        step_idx = state.step_of_day
        key = "|".join(
            [
                state.step_bin,
                state.burden,
                action,
                state.day_of_week,
                state.sleep,
            ]
        )
        entry = self._within_day[step_idx].get(key, {})
        probs = entry.get("_", {})
        expected = 0.0
        for next_step_bin, prob in probs.items():
            reward = (
                ALPHA * STEP_BIN_VALUES.get(next_step_bin, 0.0)
                + (1 - ALPHA) * SLEEP_VALUES.get(state.sleep, 0.0)
                - ACTION_PENALTIES.get(action, 0.0)
            )
            expected += prob * reward
        return expected


def _evaluate_policy(
    config,
    agent,
    label: str,
    n_seeds: int = N_SEEDS,
) -> dict:
    """Run agent over n_seeds and compute aggregate metrics."""
    all_rewards = []
    for seed in range(1, n_seeds + 1):
        records = run_episode(config, agent, seed=seed)
        all_rewards.append([r["reward"] for r in records])
    arr = np.array(all_rewards)
    return {
        "policy": label,
        "total_mean": float(arr.sum(axis=1).mean()),
        "total_std": float(arr.sum(axis=1).std()),
        "step_mean": float(arr.mean()),
    }


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        description="Compute reference bounds for sprint1_bootstrap via simulation"
    )
    parser.add_argument("config", type=str, help="Config filename in configs/")
    args = parser.parse_args()

    config_path = resolve_config(args.config)
    config = load_config(config_path)

    raw_table_dir = config.transition_model.table_dir
    if raw_table_dir is None:
        _msg = "transition_model.table_dir is not set in config"
        raise ValueError(_msg)
    table_dir = Path(raw_table_dir)
    within_day = []
    for i in range(config.steps_per_day):
        with (table_dir / f"within_day_{i}.json").open(encoding="utf-8") as f:
            within_day.append(json.load(f))

    logger.info("Config: %s", config_path)
    logger.info("Table dir: %s", table_dir)
    logger.info("Seeds per policy: %d\n", N_SEEDS)

    results = []

    # Fixed-action policies
    for a in config.action_names:
        agent = _FixedActionAgent(a)
        res = _evaluate_policy(config, agent, f"fixed_{a}")
        results.append(res)
        logger.info(
            "  %-25s total=%.1f \u00b1 %.1f  step=%.4f",
            res["policy"],
            res["total_mean"],
            res["total_std"],
            res["step_mean"],
        )

    # Random policy
    all_random_rewards = []
    for seed in range(1, N_SEEDS + 1):
        agent = make_agent(
            "random",
            actions=config.action_names,
            seed=derive_agent_seed(seed, agent_index=0),
        )
        records = run_episode(config, agent, seed=seed)
        all_random_rewards.append([r["reward"] for r in records])
    arr = np.array(all_random_rewards)
    random_res = {
        "policy": "random",
        "total_mean": float(arr.sum(axis=1).mean()),
        "total_std": float(arr.sum(axis=1).std()),
        "step_mean": float(arr.mean()),
    }
    results.append(random_res)
    logger.info(
        "  %-25s total=%.1f \u00b1 %.1f  step=%.4f",
        random_res["policy"],
        random_res["total_mean"],
        random_res["total_std"],
        random_res["step_mean"],
    )

    # Greedy oracle
    oracle_agent = _GreedyOracleAgent(within_day, config.action_names)
    oracle_res = _evaluate_policy(config, oracle_agent, "greedy_oracle")
    results.append(oracle_res)
    logger.info(
        "  %-25s total=%.1f \u00b1 %.1f  step=%.4f",
        oracle_res["policy"],
        oracle_res["total_mean"],
        oracle_res["total_std"],
        oracle_res["step_mean"],
    )

    logger.info("\nDone \u2014 evaluated %d policies.", len(results))


def compute_bounds(config):  # noqa: ARG001
    """Compatibility shim: returns None to signal bounds are unavailable.

    The sprint1_bootstrap optimal bound computation is too expensive for
    a static analysis function. Use the CLI directly:
        uv run python optimal_bound.py configs/sprint1_bootstrap.yaml
    """
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
