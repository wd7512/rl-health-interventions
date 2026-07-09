"""Analyze LLM bootstrap transition table results.

Loads one or more results JSONL files, parses responses, reconstructs
cell parameters from flat indices, and produces:
  - Text analysis (parse rates, distributions, consistency, transition realism)
  - Figures per model in subdirectories

Usage:
    uv run python scripts/analyze_bootstrap.py \\
        --files data/bootstrap/results_deepseek.jsonl \\
               data/bootstrap/results_glm5.2.jsonl \\
        --fig-dir docs/figures
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl_health_interventions.llm_bootstrapping.parse import (
    parse_day_boundary,
    parse_within_day,
)
from rl_health_interventions.llm_bootstrapping.prompts.sprint1 import (
    ACTIONS,
    BURDENS,
    DAY_TYPES,
    SLEEP_TYPES,
    STEP_BINS,
)

logger = logging.getLogger(__name__)

DAY_BOUNDARY_COUNT = 36
SAMPLES_DAY_BOUNDARY = 20
WITHIN_DAY_START = 720
WITHIN_DAY_CELLS_PER_TIMESTEP = 144
SAMPLES_WITHIN_DAY = 30
TIMESTEPS = 5

STEP_BIN_COLORS = {"inactive": "#e74c3c", "moderate": "#f39c12", "active": "#2ecc71"}
SLEEP_COLORS = {"good": "#2ecc71", "poor": "#e74c3c"}
ACTIONS_SHORT = {
    "idle": "Idle",
    "movement_suggestion": "Move",
    "goal_reminder": "Goal",
    "journal": "Journal",
}


def _day_boundary_params(cell_idx: int) -> dict:
    combos = list(itertools.product(STEP_BINS, BURDENS, DAY_TYPES, SLEEP_TYPES))
    sb, b, d, s = combos[cell_idx]
    return {"step_bin_daily": sb, "burden": b, "day_type": d, "sleep": s}


def _within_day_params(cell_idx: int) -> dict:
    combos = list(
        itertools.product(STEP_BINS, BURDENS, ACTIONS, DAY_TYPES, SLEEP_TYPES)
    )
    sb, b, a, d, s = combos[cell_idx]
    return {"step_bin": sb, "burden": b, "action": a, "day_type": d, "sleep": s}


def _parse_record(rec: dict) -> tuple[str, object]:
    idx = rec["index"]
    content = rec.get("content", "")
    if idx < WITHIN_DAY_START:
        return "day_boundary", parse_day_boundary(content)
    return "within_day", parse_within_day(content)


def _cell_info(idx: int) -> tuple[str, int, int, dict]:
    if idx < WITHIN_DAY_START:
        cell_idx = idx // SAMPLES_DAY_BOUNDARY
        sample = idx % SAMPLES_DAY_BOUNDARY
        return "day_boundary", cell_idx, sample, _day_boundary_params(cell_idx)
    offset = idx - WITHIN_DAY_START
    within_off = offset % (WITHIN_DAY_CELLS_PER_TIMESTEP * SAMPLES_WITHIN_DAY)
    cell_idx = within_off // SAMPLES_WITHIN_DAY
    sample = within_off % SAMPLES_WITHIN_DAY
    return "within_day", cell_idx, sample, _within_day_params(cell_idx)


def _load(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _model_label(path: Path) -> str:
    name = path.stem
    if name.startswith("results_"):
        return name[len("results_") :]
    return name


def _print_basic_stats(label: str, records: list[dict]) -> None:
    logger.info("")
    logger.info("%s", "=" * 60)
    logger.info("  %s: basic statistics", label)
    logger.info("%s", "=" * 60)

    total = len(records)
    day_recs = [r for r in records if r["index"] < WITHIN_DAY_START]
    within_recs = [r for r in records if r["index"] >= WITHIN_DAY_START]

    db_parsed = [r for r in day_recs if _parse_record(r)[1] is not None]
    wd_parsed = [r for r in within_recs if _parse_record(r)[1] is not None]

    logger.info("  Total records:     %s", total)
    logger.info(
        "  Day-boundary:      %s records, %s parsed", len(day_recs), len(db_parsed)
    )
    logger.info(
        "  Within-day:        %s records, %s parsed", len(within_recs), len(wd_parsed)
    )

    if day_recs:
        logger.info(
            "  Day-boundary parse rate: %.1f%%", len(db_parsed) / len(day_recs) * 100
        )
    if within_recs:
        logger.info(
            "  Within-day parse rate:   %.1f%%", len(wd_parsed) / len(within_recs) * 100
        )

    sleep_dist: dict[str, int] = defaultdict(int)
    step_bin_dist: dict[str, int] = defaultdict(int)
    step_counts: list[int] = []
    for r in records:
        cat, parsed = _parse_record(r)
        if cat == "day_boundary" and isinstance(parsed, str):
            sleep_dist[parsed] += 1
        elif cat == "within_day" and isinstance(parsed, tuple):
            steps, sb = parsed
            if isinstance(sb, str) and isinstance(steps, int):
                step_bin_dist[sb] += 1
                step_counts.append(steps)

    if sleep_dist:
        logger.info("")
        logger.info("  Sleep quality distribution:")
        for k in ("good", "poor"):
            v = sleep_dist.get(k, 0)
            total_s = sum(sleep_dist.values())
            logger.info("    %s: %s (%.1f%%)", k, v, v / total_s * 100)

    if step_bin_dist:
        logger.info("")
        logger.info("  Step bin distribution:")
        total_sb = sum(step_bin_dist.values())
        for k in ("inactive", "moderate", "active"):
            v = step_bin_dist.get(k, 0)
            logger.info("    %s: %s (%.1f%%)", k, v, v / total_sb * 100)

    if step_counts:
        arr = np.array(step_counts)
        logger.info("")
        logger.info("  Step count stats:")
        logger.info(
            "    mean=%.1f  median=%.1f  std=%.1f",
            arr.mean(),
            np.median(arr),
            arr.std(),
        )
        logger.info(
            "    p5=%.0f  p25=%.0f  p75=%.0f  p95=%.0f",
            np.percentile(arr, 5),
            np.percentile(arr, 25),
            np.percentile(arr, 75),
            np.percentile(arr, 95),
        )

    failures = [r for r in records if _parse_record(r)[1] is None]
    if failures:
        logger.info("")
        logger.info("  Parse failures (%s):", len(failures))
        for r in failures[:5]:
            content = r.get("content", "")[:150]
            logger.info("    idx=%s: %r", r["index"], content)


def _compute_cell_consistency(records: list[dict]) -> dict:
    cells: dict[str, dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for r in records:
        cat, parsed = _parse_record(r)
        if cat == "day_boundary":
            cell_idx = r["index"] // SAMPLES_DAY_BOUNDARY
            if isinstance(parsed, str):
                cells["day_boundary"][cell_idx][parsed] += 1
        elif cat == "within_day":
            cell_idx = (r["index"] - WITHIN_DAY_START) // SAMPLES_WITHIN_DAY
            if isinstance(parsed, tuple):
                _, sb = parsed
                if isinstance(sb, str):
                    cells["within_day"][cell_idx][sb] += 1
    return dict(cells)


def _print_consistency(label: str, records: list[dict]) -> None:
    logger.info("")
    logger.info("%s", "=" * 60)
    logger.info("  %s: cell consistency", label)
    logger.info("%s", "=" * 60)

    cells = _compute_cell_consistency(records)

    for cat in ("day_boundary", "within_day"):
        n_samples = (
            SAMPLES_DAY_BOUNDARY if cat == "day_boundary" else SAMPLES_WITHIN_DAY
        )
        cat_cells = cells.get(cat, {})
        if not cat_cells:
            continue
        agreements = []
        for outcomes in cat_cells.values():
            total = sum(outcomes.values())
            if total > 0:
                agreements.append(max(outcomes.values()) / total)
        arr = np.array(agreements)
        logger.info("")
        logger.info("  %s: %s cells, n=%s samples each", cat, len(cat_cells), n_samples)
        logger.info("    mean agreement: %.1f%%", arr.mean() * 100)
        logger.info("    min agreement:  %.1f%%", arr.min() * 100)
        logger.info("    100%% agreement cells: %s / %s", (arr == 1.0).sum(), len(arr))
        logger.info("    <80%% agreement cells: %s / %s", (arr < 0.8).sum(), len(arr))


def _compute_day_boundary_transitions(records: list[dict]) -> dict:
    counts: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[tuple, int] = defaultdict(int)
    for r in records:
        cat, parsed = _parse_record(r)
        if cat != "day_boundary" or not isinstance(parsed, str):
            continue
        params = _day_boundary_params(r["index"] // SAMPLES_DAY_BOUNDARY)
        key = (
            params["step_bin_daily"],
            params["burden"],
            params["day_type"],
            params["sleep"],
        )
        counts[key][parsed] += 1
        totals[key] += 1
    probs = {}
    for key, outcomes in counts.items():
        t = totals[key]
        probs[key] = {o: c / t for o, c in outcomes.items()}
    return probs


def _compute_within_day_transitions(records: list[dict]) -> dict:
    all_probs: dict[int, dict] = {}
    for t in range(TIMESTEPS):
        counts: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        totals: dict[tuple, int] = defaultdict(int)
        for r in records:
            cat, parsed = _parse_record(r)
            if cat != "within_day" or not isinstance(parsed, tuple):
                continue
            offset = r["index"] - WITHIN_DAY_START
            timestep = offset // (WITHIN_DAY_CELLS_PER_TIMESTEP * SAMPLES_WITHIN_DAY)
            if timestep != t:
                continue
            within_off = offset % (WITHIN_DAY_CELLS_PER_TIMESTEP * SAMPLES_WITHIN_DAY)
            cell_idx = within_off // SAMPLES_WITHIN_DAY
            params = _within_day_params(cell_idx)
            key = (
                params["step_bin"],
                params["burden"],
                params["action"],
                params["day_type"],
                params["sleep"],
            )
            _, sb = parsed
            if not isinstance(sb, str):
                continue
            counts[key][sb] += 1
            totals[key] += 1
        probs = {}
        for key, outcomes in counts.items():
            tt = totals[key]
            probs[key] = {o: c / tt for o, c in outcomes.items()}
        all_probs[t] = probs
    return all_probs


def _weighted_distribution(
    probs: dict,
    keys: list[tuple],
    outcomes: tuple[str, ...],
) -> dict[str, float]:
    dist = defaultdict(float)
    total = 0.0
    for key in keys:
        outcome_probs = probs.get(key, {})
        weight = sum(outcome_probs.values())
        for outcome in outcomes:
            dist[outcome] += outcome_probs.get(outcome, 0.0) * weight
        total += weight
    if not total:
        return dict.fromkeys(outcomes, 0.0)
    return {outcome: dist[outcome] / total for outcome in outcomes}


def _parse_counts(records: list[dict]) -> tuple[tuple[int, int], tuple[int, int]]:
    day_recs = [r for r in records if r["index"] < WITHIN_DAY_START]
    within_recs = [r for r in records if r["index"] >= WITHIN_DAY_START]
    day_parsed = sum(1 for r in day_recs if _parse_record(r)[1] is not None)
    within_parsed = sum(1 for r in within_recs if _parse_record(r)[1] is not None)
    return (day_parsed, len(day_recs)), (within_parsed, len(within_recs))


def _print_day_boundary_realism(label: str, records: list[dict]) -> None:
    logger.info("")
    logger.info("%s", "=" * 60)
    logger.info("  %s: day-boundary transition realism", label)
    logger.info("%s", "=" * 60)

    probs = _compute_day_boundary_transitions(records)
    if not probs:
        logger.info("  No parsed day-boundary records.")
        return

    gc = sum(probs[k].get("good", 0) * sum(probs[k].values()) for k in probs)
    tc = sum(sum(probs[k].values()) for k in probs)
    p_good = gc / tc if tc else 0
    logger.info("")
    logger.info("  Marginal P(good sleep): %.1f%%", p_good * 100)

    for sb in STEP_BINS:
        keys = [k for k in probs if k[0] == sb]
        if not keys:
            continue
        gc = sum(probs[k].get("good", 0) * sum(probs[k].values()) for k in keys)
        tc = sum(sum(probs[k].values()) for k in keys)
        val = gc / tc if tc else 0
        logger.info("  P(good | step_bin_daily=%s): %.1f%%", sb, val * 100)

    for b in BURDENS:
        keys = [k for k in probs if k[1] == b]
        if not keys:
            continue
        gc = sum(probs[k].get("good", 0) * sum(probs[k].values()) for k in keys)
        tc = sum(sum(probs[k].values()) for k in keys)
        val = gc / tc if tc else 0
        logger.info("  P(good | burden=%s): %.1f%%", b, val * 100)

    for s in SLEEP_TYPES:
        keys = [k for k in probs if k[3] == s]
        if not keys:
            continue
        gc = sum(probs[k].get("good", 0) * sum(probs[k].values()) for k in keys)
        tc = sum(sum(probs[k].values()) for k in keys)
        val = gc / tc if tc else 0
        logger.info("  P(good | sleep_last=%s): %.1f%%", s, val * 100)

    for d in DAY_TYPES:
        keys = [k for k in probs if k[2] == d]
        if not keys:
            continue
        gc = sum(probs[k].get("good", 0) * sum(probs[k].values()) for k in keys)
        tc = sum(sum(probs[k].values()) for k in keys)
        val = gc / tc if tc else 0
        logger.info("  P(good | day_type=%s): %.1f%%", d, val * 100)


def _print_within_day_realism(label: str, records: list[dict]) -> None:
    logger.info("")
    logger.info("%s", "=" * 60)
    logger.info("  %s: within-day transition realism", label)
    logger.info("%s", "=" * 60)

    all_probs = _compute_within_day_transitions(records)

    for t in range(TIMESTEPS):
        probs = all_probs.get(t, {})
        if not probs:
            continue
        logger.info("")
        logger.info("  Timestep %s:", t)
        for sb in STEP_BINS:
            keys = [k for k in probs if k[0] == sb]
            if not keys:
                continue
            dist: dict[str, float] = defaultdict(float)
            total = 0
            for k in keys:
                w = sum(probs[k].values())
                for o, p in probs[k].items():
                    dist[o] += p * w
                total += w
            if total:
                parts = ", ".join(
                    f"P({o})={dist.get(o, 0) / total:.1%}" for o in STEP_BINS
                )
                logger.info("    current=%s: %s", sb, parts)

    logger.info("")
    logger.info("  Action effect (averaged across timesteps):")
    for action in ACTIONS:
        dist = defaultdict(float)
        total = 0
        for t in range(TIMESTEPS):
            for key, outcomes in all_probs.get(t, {}).items():
                if key[2] != action:
                    continue
                w = sum(outcomes.values())
                for o, p in outcomes.items():
                    dist[o] += p * w
                total += w
        if total:
            parts = ", ".join(f"{o}={dist.get(o, 0) / total:.1%}" for o in STEP_BINS)
            logger.info("    %s: %s", action, parts)

    logger.info("")
    logger.info("  Burden effect (averaged across timesteps):")
    for burden in BURDENS:
        dist = defaultdict(float)
        total = 0
        for t in range(TIMESTEPS):
            for key, outcomes in all_probs.get(t, {}).items():
                if key[1] != burden:
                    continue
                w = sum(outcomes.values())
                for o, p in outcomes.items():
                    dist[o] += p * w
                total += w
        if total:
            parts = ", ".join(f"{o}={dist.get(o, 0) / total:.1%}" for o in STEP_BINS)
            logger.info("    %s: %s", burden, parts)

    logger.info("")
    logger.info("  Sleep effect (averaged across timesteps):")
    for sleep in SLEEP_TYPES:
        dist = defaultdict(float)
        total = 0
        for t in range(TIMESTEPS):
            for key, outcomes in all_probs.get(t, {}).items():
                if key[4] != sleep:
                    continue
                w = sum(outcomes.values())
                for o, p in outcomes.items():
                    dist[o] += p * w
                total += w
        if total:
            parts = ", ".join(f"{o}={dist.get(o, 0) / total:.1%}" for o in STEP_BINS)
            logger.info("    sleep=%s: %s", sleep, parts)

    logger.info("")
    logger.info("  Monotonicity check P(active | current_bin):")
    for t in range(TIMESTEPS):
        probs = all_probs.get(t, {})
        p_active: dict[str, float] = {}
        for sb in STEP_BINS:
            keys = [k for k in probs if k[0] == sb]
            if not keys:
                continue
            gc = sum(probs[k].get("active", 0) * sum(probs[k].values()) for k in keys)
            tc = sum(sum(probs[k].values()) for k in keys)
            p_active[sb] = gc / tc if tc else 0
        vals = [p_active.get(sb, 0) for sb in STEP_BINS]
        mono = vals[0] <= vals[1] <= vals[2]
        status = "PASS" if mono else "FAIL"
        logger.info(
            "    timestep %s: %.1f%% <= %.1f%% <= %.1f%%  [%s]",
            t,
            vals[0] * 100,
            vals[1] * 100,
            vals[2] * 100,
            status,
        )


def _plot_parse_rate(records_per_model: dict[str, list[dict]], fig_dir: Path) -> None:
    models = list(records_per_model.keys())
    db_rates = []
    wd_rates = []
    db_counts = []
    wd_counts = []
    for records in records_per_model.values():
        (db_p, db_n), (wd_p, wd_n) = _parse_counts(records)
        db_rates.append(db_p / db_n if db_n else 0)
        wd_rates.append(wd_p / wd_n if wd_n else 0)
        db_counts.append((db_p, db_n))
        wd_counts.append((wd_p, wd_n))

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - w / 2, db_rates, w, label="Day-boundary", color="#3498db")
    ax.bar(x + w / 2, wd_rates, w, label="Within-day", color="#2ecc71")
    ax.set_ylabel("Parsed responses / total responses")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    for i, (day_rate, within_rate) in enumerate(zip(db_rates, wd_rates)):
        day_p, day_n = db_counts[i]
        within_p, within_n = wd_counts[i]
        ax.text(
            i - w / 2,
            day_rate + 0.02,
            f"{day_rate:.1%}\n{day_p}/{day_n}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            i + w / 2,
            within_rate + 0.02,
            f"{within_rate:.1%}\n{within_p}/{within_n}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.suptitle(
        "Parse reliability by model and prompt family",
        y=1.02,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.99,
        "Parse reliability by model. Higher = better.",
        ha="center",
        va="top",
        fontsize=6.5,
        style="italic",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "parse_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_output_dist(records_per_model: dict[str, list[dict]], fig_dir: Path) -> None:
    for label, records in records_per_model.items():
        sleep_dist: dict[str, int] = defaultdict(int)
        sb_dist: dict[str, int] = defaultdict(int)
        total_sleep = 0
        total_sb = 0
        for r in records:
            cat, parsed = _parse_record(r)
            if cat == "day_boundary" and isinstance(parsed, str):
                sleep_dist[parsed] += 1
                total_sleep += 1
            elif cat == "within_day" and isinstance(parsed, tuple):
                _, sb = parsed
                if isinstance(sb, str):
                    sb_dist[sb] += 1
                    total_sb += 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

        if sleep_dist:
            keys = ["good", "poor"]
            vals = [sleep_dist.get(k, 0) for k in keys]
            bars = ax1.bar(keys, vals, color=[SLEEP_COLORS[k] for k in keys])
            ax1.set_title(f"{label} — Sleep quality  (n={total_sleep})")
            ax1.set_ylabel("Count")
            for bar, v in zip(bars, vals):
                pct = v / total_sleep * 100
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total_sleep * 0.01,
                    f"{v:,} ({pct:.1f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            ax1.set_ylim(0, max(vals) * 1.25)

        if sb_dist:
            keys = ["inactive", "moderate", "active"]
            vals = [sb_dist.get(k, 0) for k in keys]
            colors = [STEP_BIN_COLORS[k] for k in keys]
            bars = ax2.bar(keys, vals, color=colors)
            ax2.set_title(f"{label} — Step bin  (n={total_sb})")
            ax2.set_ylabel("Count")
            for bar, v in zip(bars, vals):
                pct = v / total_sb * 100
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total_sb * 0.01,
                    f"{v:,} ({pct:.1f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            ax2.set_ylim(0, max(vals) * 1.25)

        fig.tight_layout()
        fig.text(
            0.5,
            0.99,
            "Marginal output distribution. Realistic models put mass on "
            "all three step bins and both sleep outcomes.",
            ha="center",
            va="top",
            fontsize=6.5,
            style="italic",
        )
        fig.savefig(fig_dir / "output_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_cell_consistency(
    records_per_model: dict[str, list[dict]], fig_dir: Path
) -> None:
    for label, records in records_per_model.items():
        cells = _compute_cell_consistency(records)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

        for ax, cat, _n_samp in [
            (ax1, "day_boundary", SAMPLES_DAY_BOUNDARY),
            (ax2, "within_day", SAMPLES_WITHIN_DAY),
        ]:
            cat_cells = cells.get(cat, {})
            if not cat_cells:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{label} — {cat}")
                continue
            agreements = []
            for outcomes in cat_cells.values():
                total = sum(outcomes.values())
                if total > 0:
                    agreements.append(max(outcomes.values()) / total)
            arr = np.array(agreements)

            ax.hist(arr, bins=20, edgecolor="white", alpha=0.8, color="#3498db")

            for thresh, color in [(0.8, "#e74c3c"), (0.9, "#f39c12"), (1.0, "#2ecc71")]:
                ax.axvline(thresh, color=color, linestyle=":", alpha=0.5, linewidth=1)
                below = (arr < thresh).sum()
                ax.text(
                    thresh,
                    ax.get_ylim()[1] * 0.95,
                    f"{below}",
                    ha="center",
                    va="top",
                    fontsize=7,
                    color=color,
                )

            ax.axvline(
                arr.mean(), color="red", linestyle="--", label=f"mean={arr.mean():.1%}"
            )
            ax.set_xlabel("Agreement Fraction")
            ax.set_ylabel("Number of Cells")
            ax.set_title(f"{label} — {cat.replace('_', ' ').title()} Cell Consistency")
            ax.legend(fontsize=8)

        fig.suptitle(f"{label} — Cell Consistency", y=1.02, fontweight="bold")
        fig.text(
            0.5,
            0.99,
            "Cell-level agreement: histogram of modal-answer repeat rate. "
            "Colored lines mark cells below 80%, 90%, and 100% agreement.",
            ha="center",
            va="top",
            fontsize=6.5,
            style="italic",
        )
        fig.tight_layout()
        fig.savefig(fig_dir / "cell_consistency.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_step_hist(records_per_model: dict[str, list[dict]], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    model_steps: dict[str, list[int]] = {}
    for label, records in records_per_model.items():
        steps = []
        for r in records:
            cat, parsed = _parse_record(r)
            if cat == "within_day" and isinstance(parsed, tuple):
                s, sb = parsed
                if isinstance(s, int) and isinstance(sb, str):
                    steps.append(s)
        if steps:
            model_steps[label] = steps
            ax.hist(steps, bins=50, alpha=0.5, label=label, edgecolor="white")

    if model_steps:
        ax.set_xlabel("Step Count")
        ax.set_ylabel("Frequency")
        ax.set_title("Step Count Distribution (Within-Day)")

        stats_lines = []
        for label, steps in model_steps.items():
            arr = np.array(steps)
            stats_lines.append(
                f"{label}: mean={arr.mean():.1f}  median={np.median(arr):.1f}  "
                f"p5={np.percentile(arr, 5):.0f}  p95={np.percentile(arr, 95):.0f}"
            )
            ax.axvline(arr.mean(), linestyle="--", alpha=0.6, linewidth=1)
            ax.axvline(np.median(arr), linestyle=":", alpha=0.6, linewidth=1)

        stats_text = "\n".join(stats_lines)
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.3",
                "fc": "white",
                "ec": "gray",
                "alpha": 0.8,
            },
        )
        ax.legend(fontsize=8)

    fig.suptitle("Step Count Distribution (Within-Day)", y=1.02, fontweight="bold")
    fig.text(
        0.5,
        0.99,
        "Step count distribution (within-day). Dashed = per-model means. "
        "Realistic: spread across inactive, moderate, and active ranges.",
        ha="center",
        va="top",
        fontsize=6.5,
        style="italic",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "step_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_day_boundary_heatmap(
    records_per_model: dict[str, list[dict]], fig_dir: Path
) -> None:
    for label, records in records_per_model.items():
        probs = _compute_day_boundary_transitions(records)
        if not probs:
            continue

        fig, axes = plt.subplots(
            len(SLEEP_TYPES),
            len(BURDENS),
            figsize=(16, 8.5),
            sharey=True,
            constrained_layout=True,
        )
        im = None
        for row_idx, sleep_cond in enumerate(SLEEP_TYPES):
            for col_idx, burden in enumerate(BURDENS):
                ax = axes[row_idx, col_idx]
                data = np.zeros((len(STEP_BINS), len(DAY_TYPES)))
                for i, sb in enumerate(STEP_BINS):
                    for j, d in enumerate(DAY_TYPES):
                        key = (sb, burden, d, sleep_cond)
                        data[i, j] = probs.get(key, {}).get("good", 0.0)
                im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
                ax.set_xticks(range(len(DAY_TYPES)))
                ax.set_xticklabels(DAY_TYPES)
                ax.set_yticks(range(len(STEP_BINS)))
                ax.set_yticklabels(STEP_BINS)
                if row_idx == 0:
                    ax.set_title(f"burden={burden}", fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(f"sleep={sleep_cond}", fontsize=10, fontweight="bold")
                for i in range(len(STEP_BINS)):
                    for j in range(len(DAY_TYPES)):
                        val = data[i, j]
                        color = "white" if val < 0.4 or val > 0.6 else "black"
                        ax.text(
                            j,
                            i,
                            f"{val:.0%}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color=color,
                        )

        fig.suptitle(
            f"{label} — P(next sleep = good | step_bin_daily, burden, day_type, sleep)",
            y=1.02,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.99,
            "P(next sleep=good) | step_bin_daily, day_type, burden, current sleep. "
            "Realistic: poor sleep or higher burden reduces P(good).",
            ha="center",
            va="top",
            fontsize=6.5,
            style="italic",
        )
        assert im is not None
        fig.colorbar(im, ax=axes, shrink=0.8, label="P(good next sleep)")
        fig.savefig(fig_dir / "day_boundary_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_within_day_action(
    records_per_model: dict[str, list[dict]], fig_dir: Path
) -> None:
    for label, records in records_per_model.items():
        all_probs = _compute_within_day_transitions(records)
        fig, axes = plt.subplots(
            len(STEP_BINS), TIMESTEPS, figsize=(20, 8.5), sharey="row", sharex=True
        )
        for row_idx, current_bin in enumerate(STEP_BINS):
            for col_idx, t in enumerate(range(TIMESTEPS)):
                ax = axes[row_idx, col_idx]
                probs = all_probs.get(t, {})
                x = np.arange(len(ACTIONS))
                width = 0.25
                for bar_idx, outcome_bin in enumerate(STEP_BINS):
                    vals = []
                    for action in ACTIONS:
                        keys = [
                            k for k in probs if k[0] == current_bin and k[2] == action
                        ]
                        if keys:
                            dist: dict[str, float] = defaultdict(float)
                            total = 0
                            for k in keys:
                                w = sum(probs[k].values())
                                for o, p in probs[k].items():
                                    dist[o] += p * w
                                total += w
                            vals.append(
                                dist.get(outcome_bin, 0) / total if total else 0
                            )
                        else:
                            vals.append(0)
                    ax.bar(
                        x + bar_idx * width,
                        vals,
                        width,
                        label=outcome_bin,
                        color=STEP_BIN_COLORS[outcome_bin],
                        alpha=0.8,
                    )
                ax.set_xticks(x + width)
                ax.set_xticklabels([ACTIONS_SHORT[a] for a in ACTIONS], fontsize=7)
                if row_idx == 0:
                    ax.set_title(f"t={t}", fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(
                        f"current={current_bin}", fontsize=9, fontweight="bold"
                    )
                if row_idx == len(STEP_BINS) - 1 and col_idx == TIMESTEPS - 1:
                    ax.legend(fontsize=7, loc="lower right", title="next")

        fig.suptitle(
            f"{label} — P(next step bin | current step bin, action) by timestep",
            y=1.02,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.99,
            "P(next step bin | current bin, action) per timestep. "
            "Realistic: Move boosts P(active); persistence visible.",
            ha="center",
            va="top",
            fontsize=6.5,
            style="italic",
        )
        fig.tight_layout()
        fig.savefig(fig_dir / "within_day_action.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_burden_interaction(
    records_per_model: dict[str, list[dict]], fig_dir: Path
) -> None:
    for label, records in records_per_model.items():
        all_probs = _compute_within_day_transitions(records)

        marginal_active = _marginal_active_prob(all_probs)

        fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), sharey=True)
        for ax, action in zip(axes, ACTIONS):
            for sb in STEP_BINS:
                vals = []
                for burden in BURDENS:
                    dist = defaultdict(float)
                    total = 0
                    for t in range(TIMESTEPS):
                        for key, outcomes in all_probs.get(t, {}).items():
                            if key[0] == sb and key[2] == action and key[1] == burden:
                                w = sum(outcomes.values())
                                for o, p in outcomes.items():
                                    dist[o] += p * w
                                total += w
                    vals.append(dist.get("active", 0) / total if total else 0)
                ax.plot(BURDENS, vals, marker="o", label=sb, color=STEP_BIN_COLORS[sb])
                if vals[0] > 0 and vals[-1] > 0:
                    delta = vals[-1] - vals[0]
                    ax.annotate(
                        f"{delta:+.1%}",
                        xy=(BURDENS[-1], vals[-1]),
                        xytext=(8, 4),
                        textcoords="offset points",
                        fontsize=7,
                        color=STEP_BIN_COLORS[sb],
                    )

            ax.axhline(
                marginal_active,
                color="gray",
                linestyle=":",
                alpha=0.7,
                linewidth=1,
                label=f"marginal={marginal_active:.0%}",
            )
            ax.set_xlabel("Burden Level")
            ax.set_title(f"action={action}")
            ax.set_ylim(0, 1)
            if action == ACTIONS[0]:
                ax.set_ylabel("P(active next)")
                ax.legend(fontsize=7)

        fig.suptitle(
            f"{label} — P(active next | current step bin, action, burden)",
            y=1.02,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.99,
            "P(active next) vs burden by current step bin and action. "
            "Gray dashed = marginal P(active). Higher burden should reduce P(active). "
            "Delta labels show change from low to high burden.",
            ha="center",
            va="top",
            fontsize=6.5,
            style="italic",
        )
        fig.tight_layout()
        fig.savefig(fig_dir / "burden_interaction.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _marginal_active_prob(all_probs: dict[int, dict]) -> float:
    total_active = 0.0
    total_all = 0.0
    for t_probs in all_probs.values():
        for outcomes in t_probs.values():
            w = sum(outcomes.values())
            total_active += outcomes.get("active", 0) * w
            total_all += w
    return total_active / total_all if total_all else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Results JSONL files to analyze",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Base directory for figure output (default: docs/figures)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model_data: dict[str, list[dict]] = {}
    for fpath in args.files:
        path = Path(fpath)
        label = _model_label(path)
        logger.info("Loading %s (%s)...", path, label)
        model_data[label] = _load(path)

    for label, records in model_data.items():
        out_dir = args.fig_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)

        _print_basic_stats(label, records)
        _print_consistency(label, records)
        _print_day_boundary_realism(label, records)
        _print_within_day_realism(label, records)

        _plot_parse_rate(model_data, out_dir)
        _plot_output_dist({label: records}, out_dir)
        _plot_cell_consistency({label: records}, out_dir)
        _plot_step_hist(model_data, out_dir)
        _plot_day_boundary_heatmap({label: records}, out_dir)
        _plot_within_day_action({label: records}, out_dir)
        _plot_burden_interaction({label: records}, out_dir)

        logger.info("")
        logger.info("  Figures saved to %s/", out_dir)

    logger.info("")
    logger.info("%s", "=" * 60)
    logger.info("  Done.")
    logger.info("%s", "=" * 60)


if __name__ == "__main__":
    main()
