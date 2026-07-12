"""HeartSteps V2 vs bandits vs optimal DP across all personas (50 seeds)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from _shared import agent_label, resolve_config, run_agent

from rl_health_interventions.agents import derive_agent_seed
from rl_health_interventions.agents import make as make_agent
from rl_health_interventions.agents.fixed import FixedAgent
from rl_health_interventions.agents.heartsteps.agent import HeartStepsAgent
from rl_health_interventions.config.loader import load_config
from rl_health_interventions.episode import run_episode

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_CROSS = "cross"
_EXT = "sprint1_bootstrap_extensions.yaml"

_PERSONA_CONFIGS = [
    ("base", "sprint1_bootstrap.yaml"),
    ("base_masked", "sprint1_bootstrap_masked.yaml"),
    ("goal_driven", f"{_CROSS}/persona_goal_driven/{_EXT}"),
    ("resistant", f"{_CROSS}/persona_resistant/{_EXT}"),
    ("social_resp", f"{_CROSS}/persona_social_responder/{_EXT}"),
    ("stable_maint", f"{_CROSS}/persona_stable_maintainer/{_EXT}"),
    ("init_deepseek", f"{_CROSS}/initial_deepseek/{_EXT}"),
    ("init_glm5.2", f"{_CROSS}/initial_glm5.2/{_EXT}"),
]


def _summarize(rewards: np.ndarray) -> dict:
    return {
        "total_mean": float(rewards.sum(axis=1).mean()),
        "total_std": float(rewards.sum(axis=1).std()),
        "per_step": float(rewards.mean()),
        "last50": float(rewards[:, -50:].mean()),
    }


def _run_heartsteps(config, n_seeds: int, gamma: float = 0.5) -> np.ndarray:
    state_vars = {k: v.names for k, v in config.state.variables.items()}
    extra = {"step_of_day": list(range(config.steps_per_day))}
    rewards = []
    for seed in range(1, n_seeds + 1):
        agent = make_agent(
            "heartsteps",
            actions=config.action_names,
            seed=derive_agent_seed(seed, agent_index=0),
            gamma=gamma,
        )
        assert isinstance(agent, HeartStepsAgent)
        agent.init_one_hot_map(state_vars, extra_features=extra)
        records = run_episode(config, agent, seed=seed)
        rewards.append([r["reward"] for r in records])
    return np.array(rewards)


def _run_dp_bound(config_path: str) -> dict:
    # Local import: _optimal_dp lives alongside this script, not on sys.path.
    _script_dir = str(Path(__file__).resolve().parent)
    _orig_path = sys.path[:]
    sys.path.insert(0, _script_dir)
    try:
        from _optimal_dp import OptimalBound  # noqa: PLC0415
    finally:
        sys.path[:] = _orig_path

    t0 = time.perf_counter()
    bound = OptimalBound(config_path)
    bound.run()
    t1 = time.perf_counter()
    r = bound.report()
    g = bound.greedy_oracle(seeds=50)
    t2 = time.perf_counter()
    logger.info(
        "  DP: %.2f (%.4f/step) [%.1fs + %.1fs]",
        r["optimal_value"],
        r["per_step"],
        t1 - t0,
        t2 - t1,
    )
    return {
        "optimal_value": r["optimal_value"],
        "per_step": r["per_step"],
        "n_states": r["n_states"],
        "n_timesteps": r["n_timesteps"],
        "greedy_oracle_mean": g["mean"],
        "greedy_oracle_std": g["std"],
    }


def _run_idle(config, n_seeds: int) -> dict:
    rewards = []
    for seed in range(1, n_seeds + 1):
        agent = FixedAgent(action="idle")
        records = run_episode(config, agent, seed=seed)
        rewards.append([r["reward"] for r in records])
    return _summarize(np.array(rewards))


def _print_comparison(result: dict) -> None:
    dp_val = result["dp"]["optimal_value"]
    hs = result["heartsteps"]
    idle = result["idle"]
    hs_pct = 100 * hs["total_mean"] / dp_val if dp_val != 0 else 0.0
    logger.info(
        "\n  DP: %.2f | HS: %.2f (%.1f%%, %+.2f vs idle) | Idle: %.2f",
        dp_val,
        hs["total_mean"],
        hs_pct,
        hs["total_mean"] - idle["total_mean"],
        idle["total_mean"],
    )
    for al, bv in sorted(result["bandits"].items(), key=lambda x: -x[1]["total_mean"]):
        pct = 100 * bv["total_mean"] / dp_val if dp_val != 0 else 0.0
        logger.info("  %-20s %.2f (%.1f%% of DP)", al, bv["total_mean"], pct)


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        description="HeartSteps V2 benchmark across personas",
    )
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--persona", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n_seeds = args.seeds

    if args.config:
        personas = [(args.persona or "custom", args.config)]
    else:
        personas = _PERSONA_CONFIGS

    all_results = {}
    for label, cfg_rel in personas:
        config_path = resolve_config(cfg_rel)
        config = load_config(config_path)
        logger.info("\n=== %s (%d seeds) ===", label, n_seeds)

        result: dict = {"config": cfg_rel, "seeds": n_seeds}

        logger.info("DP bound...")
        result["dp"] = _run_dp_bound(config_path)

        logger.info("Bandits...")
        bandits: dict[str, dict] = {}
        for agent_cfg in config.agents:
            al = agent_label(agent_cfg)
            logger.info("  %s...", al)
            rewards = run_agent(config, agent_cfg, n_seeds)
            bandits[al] = _summarize(rewards)
        result["bandits"] = bandits

        logger.info("HeartSteps (gamma=%.1f)...", args.gamma)
        t0 = time.perf_counter()
        hs_rewards = _run_heartsteps(config, n_seeds, gamma=args.gamma)
        result["heartsteps"] = _summarize(hs_rewards)
        result["heartsteps"]["gamma"] = args.gamma
        result["heartsteps"]["time_s"] = time.perf_counter() - t0

        logger.info("Always-idle...")
        result["idle"] = _run_idle(config, n_seeds)

        _print_comparison(result)
        all_results[label] = result

    if args.config or args.persona:
        label = args.persona or Path(args.config).stem
        out_path = _RESULTS_DIR / f"heartsteps_benchmark_{label}.json"
    else:
        out_path = _RESULTS_DIR / "heartsteps_benchmark_50seeds.json"
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        f.write("\n")
    logger.info("\nWrote: %s", out_path)


if __name__ == "__main__":
    main()
