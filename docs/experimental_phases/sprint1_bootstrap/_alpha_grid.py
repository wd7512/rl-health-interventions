"""Grid search over alpha values for all 7 table sets.

For each (table_set, alpha) combination:
  1. Compute optimal DP policy via backward induction
  2. Simulate one trajectory to measure activity level
  3. Record optimal value, per-step value, and activity %

Output: JSON results + activity chart.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from _optimal_dp import OptimalBound

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
BASE_CONFIG = str(CONFIGS_DIR / "sprint1_bootstrap_extensions.yaml")

TABLE_SETS: dict[str, str] = {
    "Base": "tables/persona/base_deepseek-v4-flash",
    "Goal-driven": "tables/persona/goal_driven_deepseek-v4-flash",
    "Resistant": "tables/persona/resistant_deepseek-v4-flash",
    "Social Responder": "tables/persona/social_responder_deepseek-v4-flash",
    "Stable Maintainer": "tables/persona/stable_maintainer_deepseek-v4-flash",
    "Initial DeepSeek": "tables/initial/deepseek",
    "Initial GLM 5.2": "tables/initial/glm5.2",
}

ALPHAS = [round(i * 0.1, 1) for i in range(11)]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHART_PATH = Path(__file__).resolve().parent / "images" / "alpha_grid_activity.png"


def _resolve_table_dir(rel_path: str) -> str:
    return str((PROJECT_ROOT / rel_path).resolve())


def run_alpha_grid() -> dict:
    results: dict = {}
    for label, rel_path in TABLE_SETS.items():
        logger.info("=" * 60)
        logger.info("Table set: %s", label)
        logger.info("-" * 60)
        table_dir = _resolve_table_dir(rel_path)
        set_results: list[dict] = []
        for alpha in ALPHAS:
            bound = OptimalBound(BASE_CONFIG, alpha=alpha, table_dir=table_dir)
            r = bound.report()
            logger.info("  alpha=%.1f: solving %d states x %d timesteps...", alpha, r["n_states"], r["n_timesteps"])
            bound.run()
            r = bound.report()
            activity = bound.policy_activity(seed=42)
            entry = {
                "alpha": alpha,
                "optimal_value": r["optimal_value"],
                "per_step": r["per_step"],
                "activity_pct": activity["activity_pct"],
                "idle_pct": activity["idle_pct"],
                "n_steps": activity["n_steps"],
                "action_distribution": activity["action_distribution"],
            }
            set_results.append(entry)
            logger.info(
                "    value=%.2f (%.4f/step)  idle=%.1f%%  active=%.1f%%",
                entry["optimal_value"],
                entry["per_step"],
                entry["idle_pct"],
                entry["activity_pct"],
            )
        results[label] = set_results
    return results


def save_results(results: dict) -> None:
    out_path = RESULTS_DIR / "alpha_grid_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", out_path)


def draw_chart(results: dict) -> None:
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = matplotlib.colormaps["tab10"]
    colors = cmap(np.linspace(0, 1, len(results)))
    for (label, data), color in zip(results.items(), colors):
        alphas = [e["alpha"] for e in data]
        activity = [e["activity_pct"] for e in data]
        ax.plot(alphas, activity, marker="o", color=color, label=label, linewidth=1.5, markersize=4)

    ax.set_xlabel("Alpha (step_bin weight)", fontsize=12)
    ax.set_ylabel("Optimal Policy Activity (%)", fontsize=12)
    ax.set_title("Optimal Policy Activity vs Step-Bin Reward Weight", fontsize=14)
    ax.set_xticks(ALPHAS)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 102)
    fig.tight_layout()
    fig.savefig(str(CHART_PATH), dpi=150)
    logger.info("Chart saved to %s", CHART_PATH)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Alpha grid search over %d table sets x %d alpha values", len(TABLE_SETS), len(ALPHAS))
    results = run_alpha_grid()
    save_results(results)
    draw_chart(results)


if __name__ == "__main__":
    main()
