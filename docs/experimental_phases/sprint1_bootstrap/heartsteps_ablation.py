"""HeartSteps V2 ablation study: proxy, exploration, learning curves, sensitivity."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from _shared import resolve_config

from rl_health_interventions.agents import derive_agent_seed
from rl_health_interventions.agents import make as make_agent
from rl_health_interventions.agents.heartsteps.agent import HeartStepsAgent
from rl_health_interventions.config.loader import load_config
from rl_health_interventions.episode import run_episode

logger = logging.getLogger(__name__)
_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _summarize(rewards: np.ndarray) -> dict:
    return {
        "total_mean": float(rewards.sum(axis=1).mean()),
        "total_std": float(rewards.sum(axis=1).std()),
        "per_step": float(rewards.mean()),
    }


def _run_variant(
    config, n_seeds: int, *, gamma: float = 0.5, greedy: bool = False, use_proxy: bool = True
) -> dict:
    state_vars = {k: v.names for k, v in config.state.variables.items()}
    extra = {"step_of_day": list(range(config.steps_per_day))}
    all_rewards, all_daily = [], []
    for seed in range(1, n_seeds + 1):
        agent = make_agent(
            "heartsteps", actions=config.action_names,
            seed=derive_agent_seed(seed, agent_index=0),
            gamma=gamma, greedy=greedy, use_proxy=use_proxy,
        )
        assert isinstance(agent, HeartStepsAgent)
        agent.init_one_hot_map(state_vars, extra_features=extra)
        records = run_episode(config, agent, seed=seed)
        rewards = [r["reward"] for r in records]
        all_rewards.append(rewards)
        daily = [np.mean(rewards[d*config.steps_per_day:(d+1)*config.steps_per_day])
                 for d in range(config.episode_days)]
        all_daily.append(daily)
    rewards_arr = np.array(all_rewards)
    daily_arr = np.array(all_daily)
    s = _summarize(rewards_arr)
    s["daily_mean"] = daily_arr.mean(axis=0).tolist()
    s["daily_std"] = daily_arr.std(axis=0).tolist()
    n_days = config.episode_days
    s["early_10pct_mean"] = float(daily_arr[:, :int(n_days*0.1)].mean())
    s["late_10pct_mean"] = float(daily_arr[:, int(n_days*0.9):].mean())
    return s


def _run_dp_bound(config_path: str) -> float:
    import sys
    _script_dir = str(Path(__file__).resolve().parent)
    sys.path.insert(0, _script_dir)
    try:
        from _optimal_dp import OptimalBound
    finally:
        sys.path.pop(0)
    bound = OptimalBound(config_path)
    bound.run()
    return bound.report()["optimal_value"]


def _sweep_param(config, n_seeds: int, param: str, values: list, base_kw: dict) -> dict:
    out = {}
    for v in values:
        state_vars = {k: v_.names for k, v_ in config.state.variables.items()}
        extra = {"step_of_day": list(range(config.steps_per_day))}
        rewards = []
        for seed in range(1, n_seeds + 1):
            kw = {**base_kw, "seed": derive_agent_seed(seed, agent_index=0), param: v}
            agent = make_agent("heartsteps", actions=config.action_names, **kw)
            assert isinstance(agent, HeartStepsAgent)
            agent.init_one_hot_map(state_vars, extra_features=extra)
            records = run_episode(config, agent, seed=seed)
            rewards.append([r["reward"] for r in records])
        s = _summarize(np.array(rewards))
        out[str(v)] = s["total_mean"]
        logger.info("  %s=%s: %.2f", param, v, s["total_mean"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="HeartSteps V2 ablation study")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n_seeds = args.seeds

    config_path = resolve_config(args.config)
    config = load_config(config_path)
    dp_val = _run_dp_bound(config_path)
    logger.info("DP bound: %.2f", dp_val)
    results: dict = {"dp_value": dp_val, "seeds": n_seeds}

    # 1: Proxy ablation
    logger.info("\n=== Proxy ablation ===")
    with_p = _run_variant(config, n_seeds, gamma=0.5, use_proxy=True)
    without_p = _run_variant(config, n_seeds, gamma=0.5, use_proxy=False)
    logger.info("  With proxy: %.2f (%.1f%% DP)", with_p["total_mean"], 100*with_p["total_mean"]/dp_val)
    logger.info("  Without proxy: %.2f (%.1f%% DP)", without_p["total_mean"], 100*without_p["total_mean"]/dp_val)
    results["proxy"] = {"with": with_p, "without": without_p}

    # 2: Thompson Sampling vs greedy
    logger.info("\n=== Exploration ablation ===")
    ts = _run_variant(config, n_seeds, gamma=0.5, greedy=False)
    gr = _run_variant(config, n_seeds, gamma=0.5, greedy=True)
    logger.info("  TS: %.2f (%.1f%% DP)", ts["total_mean"], 100*ts["total_mean"]/dp_val)
    logger.info("  Greedy: %.2f (%.1f%% DP)", gr["total_mean"], 100*gr["total_mean"]/dp_val)
    results["exploration"] = {"thompson": ts, "greedy": gr}

    # 3: Sensitivity
    logger.info("\n=== Sensitivity analysis ===")
    base_kw = {"gamma": 0.5}
    results["sensitivity"] = {
        "gamma": _sweep_param(config, n_seeds, "gamma", [0.3, 0.5, 0.7, 0.9], base_kw),
        "lambda_dosage": _sweep_param(config, n_seeds, "lambda_dosage", [0.85, 0.90, 0.95, 0.99], base_kw),
        "w": _sweep_param(config, n_seeds, "w", [0.1, 0.3, 0.5, 0.7, 1.0], base_kw),
    }

    out_path = _RESULTS_DIR / "heartsteps_ablation.json"
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    logger.info("\nWrote: %s", out_path)


if __name__ == "__main__":
    main()
