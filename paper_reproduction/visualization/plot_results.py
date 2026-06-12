"""Figure 2 equivalent: per-participant improvement bar chart.

Generates a bar chart of improvement (proposed algorithm reward minus
TS Bandit reward) for each participant, sorted by improvement, with a
horizontal line at zero. Also prints a summary table to the log.

Reference:
    Liao et al. (2019). arXiv:1909.03539, Figure 2.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_results_data(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return dict(json.load(f))


def extract_improvements(data: dict[str, Any]) -> list[dict[str, Any]]:
    participants: list[dict[str, Any]] = []
    for cv in data["cv_results"]:
        for pr in cv["participant_results"]:
            participants.append(
                {
                    "participant_idx": pr["participant_idx"],
                    "improvement": pr["improvement"],
                    "proposed_mean": pr["proposed_mean"],
                    "ts_bandit_mean": pr["ts_bandit_mean"],
                }
            )
    return participants


def plot_improvement_bar_chart(
    participants: list[dict[str, Any]],
    save_path: str | Path | None = None,
    title: str = "Per-Participant Improvement: Proposed vs TS Bandit",
) -> Path:
    if save_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = RESULTS_DIR / "improvement_bar_chart.png"

    sorted_parts = sorted(participants, key=lambda p: p["improvement"])
    indices = [p["participant_idx"] for p in sorted_parts]
    improvements = [p["improvement"] for p in sorted_parts]

    colors = ["green" if imp > 0 else "red" for imp in improvements]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        range(len(improvements)),
        improvements,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(y=0, color="black", linewidth=0.8)

    ax.set_xlabel("Participant")
    ax.set_ylabel("Improvement (Total Reward)")
    ax.set_title(title)
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([str(i) for i in indices], rotation=45)

    for bar, imp in zip(bars, improvements):
        y_pos = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{imp:.1f}",
            ha="center",
            va="bottom" if y_pos >= 0 else "top",
            fontsize=8,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)

    logger.info("Bar chart saved to %s", save_path)
    return Path(save_path)


def print_summary_table(data: dict[str, Any]) -> dict[str, Any]:
    summary = data["summary"]

    logger.info("=" * 50)
    logger.info("SIMULATION RESULTS SUMMARY")
    logger.info("=" * 50)
    logger.info("Participants:     %d", summary["n_participants"])
    logger.info(
        "Improved:         %d / %d (%.1f%%)",
        summary["n_improved"],
        summary["n_participants"],
        summary["pct_improved"],
    )
    logger.info("Mean improvement: %.2f", summary["mean_improvement"])
    logger.info("Median improvement: %.2f", summary["median_improvement"])
    logger.info("Std improvement:  %.2f", summary["std_improvement"])
    logger.info("Min improvement:  %.2f", summary["min_improvement"])
    logger.info("Max improvement:  %.2f", summary["max_improvement"])
    logger.info("=" * 50)

    return summary


def generate_all_plots(
    results_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_results_data(results_path)
    participants = extract_improvements(data)
    print_summary_table(data)

    bar_path = plot_improvement_bar_chart(
        participants,
        save_path=output_dir / "improvement_bar_chart.png",
    )

    return {"improvement_bar_chart": bar_path}
