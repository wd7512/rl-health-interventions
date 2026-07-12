"""HeartSteps V2 internal analysis: β posteriors, H table, dosage, Q-values."""

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
from rl_health_interventions.environment import Environment
from rl_health_interventions.episode import run_episode

logger = logging.getLogger(__name__)
_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _build_init_vec(config, n_features: int) -> np.ndarray:
    state_vars = {k: v.names for k, v in config.state.variables.items()}
    init = config.initial_state
    vec = np.zeros(n_features, dtype=np.float64)
    offset = 0
    for var_name, vals in state_vars.items():
        for i, val in enumerate(vals):
            if init.get(var_name) == val:
                vec[offset + i] = 1.0
        offset += len(vals)
    # step_of_day: init is always step 0
    vec[offset] = 1.0
    return vec


def _track_episode(config, agent, seed: int, init_vec: np.ndarray, gamma: float) -> dict:
    state_vars = {k: v.names for k, v in config.state.variables.items()}
    extra = {"step_of_day": list(range(config.steps_per_day))}
    records = run_episode(config, agent, seed=seed)

    env = Environment(config, seed=derive_agent_seed(seed, agent_index=0))
    state = env.reset()
    agent2 = make_agent(
        "heartsteps",
        actions=config.action_names,
        seed=derive_agent_seed(seed, agent_index=0),
        gamma=gamma,
    )
    assert isinstance(agent2, HeartStepsAgent)
    agent2.init_one_hot_map(state_vars, extra_features=extra)

    betas, dosages, h_means, q_inits = [], [], [], []
    for rec in records:
        if agent2._regression is not None:
            bm = agent2.get_beta_means()
            h = agent2.get_proxy_table()
            d = agent2.get_dosage()
            qr = agent2._regression.get_reward_means(avg_features=init_vec)
            qv = {
                a: float(qr[a] + agent2.get_gamma() * agent2._proxy.get_eta(a, d))
                for a in config.action_names
            }
            betas.append(
                {
                    "day": rec["day"],
                    "step": rec["step_of_day"],
                    "betas": {
                        a: bm[a].tolist()
                        for a in agent2._regression.non_reference_actions
                    },
                }
            )
            dosages.append({"day": rec["day"], "step": rec["step_of_day"], "dosage": d})
            h_means.append(
                {
                    "day": rec["day"],
                    "h_mean": {
                        a: float(h[:, i].mean())
                        for a, i in agent2.get_action_index().items()
                    },
                }
            )
            q_inits.append(
                {"day": rec["day"], "step": rec["step_of_day"], "q_values": qv}
            )

        next_state, reward, _done = env.step(rec["action"])
        agent2.update(state, rec["action"], reward, next_state)
        if next_state.step_of_day == 0 and next_state.day > 0:
            agent2.on_day_end()
        state = next_state

    return {
        "records": records,
        "betas": betas,
        "dosages": dosages,
        "h_means": h_means,
        "q_inits": q_inits,
    }


def _run_with_tracking(config, n_seeds: int, gamma: float = 0.5) -> dict:
    state_vars = {k: v.names for k, v in config.state.variables.items()}
    extra = {"step_of_day": list(range(config.steps_per_day))}
    n_features = sum(len(v) for v in state_vars.values()) + config.steps_per_day
    feature_names = [f"{var}_{val}" for var, vals in state_vars.items() for val in vals]
    feature_names.extend(f"step_of_day_{i}" for i in range(config.steps_per_day))
    init_vec = _build_init_vec(config, n_features)

    all_seeds = []
    for si in range(1, n_seeds + 1):
        agent = make_agent(
            "heartsteps",
            actions=config.action_names,
            seed=derive_agent_seed(si, agent_index=0),
            gamma=gamma,
        )
        assert isinstance(agent, HeartStepsAgent)
        agent.init_one_hot_map(state_vars, extra_features=extra)
        all_seeds.append(_track_episode(config, agent, si, init_vec, gamma))
        if si % 10 == 0:
            logger.info("  seed %d/%d done", si, n_seeds)

    last_betas = all_seeds[0]["betas"][-1]["betas"]
    feature_importance = {}
    for i, fname in enumerate(feature_names):
        imp = np.mean([abs(last_betas[a][i]) for a in last_betas])
        feature_importance[fname] = float(imp)

    q_day0 = {
        a: float(np.mean([s["q_inits"][0]["q_values"][a] for s in all_seeds]))
        for a in config.action_names
    }
    q_daylast = {
        a: float(np.mean([s["q_inits"][-1]["q_values"][a] for s in all_seeds]))
        for a in config.action_names
    }

    beta_day0 = all_seeds[0]["betas"][0]["betas"]
    beta_daylast = all_seeds[0]["betas"][-1]["betas"]

    last_dosages = [s["dosages"][-1]["dosage"] for s in all_seeds]
    last_h = {
        a: float(np.mean([s["h_means"][-1]["h_mean"][a] for s in all_seeds]))
        for a in config.action_names
    }

    return {
        "feature_names": feature_names,
        "n_features": n_features,
        "feature_importance": dict(
            sorted(feature_importance.items(), key=lambda x: -x[1])
        ),
        "beta_day0": beta_day0,
        "beta_daylast": beta_daylast,
        "final_dosage_mean": float(np.mean(last_dosages)),
        "final_dosage_std": float(np.std(last_dosages)),
        "final_h_mean": last_h,
        "q_values_day0": q_day0,
        "q_values_daylast": q_daylast,
    }


def main() -> None:  # noqa: PLR0915, C901
    parser = argparse.ArgumentParser(description="HeartSteps V2 internal analysis")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--label", type=str, default="analysis")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config(resolve_config(args.config))
    logger.info(
        "Running HeartSteps analysis: %d seeds, gamma=%.1f", args.seeds, args.gamma
    )
    t0 = time.perf_counter()
    result = _run_with_tracking(config, args.seeds, gamma=args.gamma)
    logger.info("Done in %.1fs", time.perf_counter() - t0)

    logger.info("\n=== Feature Importance (mean |β|) ===")
    for fname, imp in result["feature_importance"].items():
        logger.info("  %-25s %.4f", fname, imp)

    logger.info("\n=== β Posterior Means (day 0 vs day 89) ===")
    for a in result["beta_daylast"]:
        logger.info("  %s:", a)
        for i, fn in enumerate(result["feature_names"]):
            d0 = result["beta_day0"][a][i]
            d89 = result["beta_daylast"][a][i]
            logger.info("    %-25s %+.4f -> %+.4f", fn, d0, d89)

    logger.info("\n=== Q-values at Initial State ===")
    logger.info("  Day 0:")
    for a, q in sorted(result["q_values_day0"].items(), key=lambda x: -x[1]):
        logger.info("    %-25s %+.4f", a, q)
    logger.info("  Day 89:")
    for a, q in sorted(result["q_values_daylast"].items(), key=lambda x: -x[1]):
        logger.info("    %-25s %+.4f", a, q)

    logger.info("\n=== Proxy H Table (mean across dosage bins) ===")
    for a, h in sorted(result["final_h_mean"].items(), key=lambda x: -x[1]):
        logger.info("  %-25s H_mean=%+.4f", a, h)

    logger.info("\n=== Dosage (day 89) ===")
    logger.info(
        "  Mean=%.4f, Std=%.4f", result["final_dosage_mean"], result["final_dosage_std"]
    )

    out_path = _RESULTS_DIR / f"heartsteps_{args.label}_analysis.json"
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    logger.info("\nWrote: %s", out_path)


if __name__ == "__main__":
    main()
