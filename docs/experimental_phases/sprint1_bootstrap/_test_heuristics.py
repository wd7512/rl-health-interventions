"""Test non-idle heuristic strategies through the real environment."""

from __future__ import annotations

import logging

import numpy as np

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.environment import Environment

logger = logging.getLogger(__name__)

_ACTION_NAMES = ["idle", "movement_suggestion", "goal_reminder", "journal"]
_IDLE_ONLY = ["idle"] * 5
_MORNING_MOVE = ["movement_suggestion", "idle", "idle", "idle", "idle"]
_MORNING_GOAL = ["goal_reminder", "idle", "idle", "idle", "idle"]
_MORNING_JOURNAL = ["journal", "idle", "idle", "idle", "idle"]
_TWO_STEP_MOVE = ["movement_suggestion", "idle", "movement_suggestion", "idle", "idle"]
_ALL_MOVE = ["movement_suggestion"] * 5
_MORNING_ROTATION = ["movement_suggestion", "idle", "idle", "idle", "idle"]
_STEP04_MOVE = ["movement_suggestion", "idle", "idle", "idle", "movement_suggestion"]


def _run_fixed(config, daily_pattern: list[str], n_seeds: int) -> dict:
    totals = []
    for s in range(n_seeds):
        env = Environment(config, seed=s)
        state = env.reset()
        ep = 0.0
        done = False
        for _ in range(config.episode_days):
            for step in range(config.steps_per_day):
                a = daily_pattern[step]
                state, reward, done = env.step(a)
                ep += reward
                if done:
                    break
            if done:
                break
        totals.append(ep)
    a = np.array(totals)
    return {"mean": float(a.mean()), "std": float(a.std()), "n_seeds": n_seeds}


def _run_conditional(config, policy_fn, n_seeds: int, label: str) -> dict:
    totals = []
    for s in range(n_seeds):
        env = Environment(config, seed=s)
        state = env.reset()
        ep = 0.0
        done = False
        while not done:
            for step_of_day in range(config.steps_per_day):
                a = policy_fn(state, step_of_day)
                state, reward, done = env.step(a)
                ep += reward
                if done:
                    break
        totals.append(ep)
    a = np.array(totals)
    logger.info("%-30s %.2f +- %.2f (%.4f/step)", label, a.mean(), a.std(), a.mean() / (config.episode_days * config.steps_per_day))
    return {"mean": float(a.mean()), "std": float(a.std()), "n_seeds": n_seeds}


def main() -> None:
    config = load_config("docs/experimental_phases/sprint1_bootstrap/configs/sprint1_bootstrap_extensions.yaml")
    n_seeds = 50
    T = config.episode_days * config.steps_per_day

    logger.info("%-30s %s", "Strategy", f"Total Reward ({n_seeds} seeds)")
    logger.info("-" * 70)

    strategies = [
        ("Always idle", _IDLE_ONLY),
        ("Morning: movement_suggestion", _MORNING_MOVE),
        ("Morning: goal_reminder", _MORNING_GOAL),
        ("Morning: journal", _MORNING_JOURNAL),
        ("Steps 0+2: movement_suggestion", _TWO_STEP_MOVE),
        ("Steps 0+4: movement_suggestion", _STEP04_MOVE),
        ("Always: movement_suggestion", _ALL_MOVE),
    ]

    for label, pat in strategies:
        r = _run_fixed(config, pat, n_seeds)
        logger.info("%-30s %.2f +- %.2f (%.4f/step)", label, r["mean"], r["std"], r["mean"] / T)

    logger.info("")
    logger.info("Conditional strategies:")
    logger.info("-" * 70)

    def conditional(label, fn):
        return _run_conditional(config, fn, n_seeds, label)

    conditional("Morning if sleep poor", lambda s, sod: "movement_suggestion" if sod == 0 and str(s.sleep) == "poor" else "idle")
    conditional("Morning if burden low", lambda s, sod: "movement_suggestion" if sod == 0 and str(s.burden) == "low" else "idle")
    conditional("Morning if inactive", lambda s, sod: "movement_suggestion" if sod == 0 and str(s.step_bin) == "inactive" else "idle")
    conditional("Morning if active (anti)", lambda s, sod: "movement_suggestion" if sod == 0 and str(s.step_bin) == "active" else "idle")
    conditional("All steps if sleep poor", lambda s, sod: "movement_suggestion" if str(s.sleep) == "poor" else "idle")
    conditional("Morning rotation (move/goal/journal)", lambda s, sod: ["movement_suggestion", "goal_reminder", "journal"][s.day % 3] if sod == 0 else "idle")
    conditional("Morning move + afternoon if inactive", lambda s, sod: "movement_suggestion" if (sod == 0 or (sod in (2, 3) and str(s.step_bin) == "inactive")) else "idle")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
