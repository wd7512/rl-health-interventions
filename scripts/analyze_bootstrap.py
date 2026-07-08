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
    with open(path) as fh:
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
    print(f"\n{'=' * 60}")
    print(f"  {label}: basic statistics")
    print(f"{'=' * 60}")

    total = len(records)
    day_recs = [r for r in records if r["index"] < WITHIN_DAY_START]
    within_recs = [r for r in records if r["index"] >= WITHIN_DAY_START]

    db_parsed = [r for r in day_recs if _parse_record(r)[1] is not None]
    wd_parsed = [r for r in within_recs if _parse_record(r)[1] is not None]

    print(f"  Total records:     {total}")
    print(f"  Day-boundary:      {len(day_recs)} records, {len(db_parsed)} parsed")
    print(f"  Within-day:        {len(within_recs)} records, {len(wd_parsed)} parsed")

    if day_recs:
        print(f"  Day-boundary parse rate: {len(db_parsed) / len(day_recs):.1%}")
    if within_recs:
        print(f"  Within-day parse rate:   {len(wd_parsed) / len(within_recs):.1%}")

    sleep_dist: dict[str, int] = defaultdict(int)
    step_bin_dist: dict[str, int] = defaultdict(int)
    step_counts: list[int] = []
    for r in records:
        cat, parsed = _parse_record(r)
        if cat == "day_boundary" and parsed is not None:
            sleep_dist[parsed] += 1
        elif cat == "within_day" and parsed is not None:
            steps, sb = parsed
            step_bin_dist[sb] += 1
            step_counts.append(steps)

    if sleep_dist:
        print("\n  Sleep quality distribution:")
        for k in ("good", "poor"):
            v = sleep_dist.get(k, 0)
            total_s = sum(sleep_dist.values())
            print(f"    {k}: {v} ({v / total_s:.1%})")

    if step_bin_dist:
        print("\n  Step bin distribution:")
        total_sb = sum(step_bin_dist.values())
        for k in ("inactive", "moderate", "active"):
            v = step_bin_dist.get(k, 0)
            print(f"    {k}: {v} ({v / total_sb:.1%})")

    if step_counts:
        arr = np.array(step_counts)
        print("\n  Step count stats:")
        print(
            f"    mean={arr.mean():.1f}  "
            f"median={np.median(arr):.1f}  std={arr.std():.1f}"
        )
        print(
            f"    p5={np.percentile(arr, 5):.0f}  p25={np.percentile(arr, 25):.0f}"
            f"  p75={np.percentile(arr, 75):.0f}  p95={np.percentile(arr, 95):.0f}"
        )

    failures = [r for r in records if _parse_record(r)[1] is None]
    if failures:
        print(f"\n  Parse failures ({len(failures)}):")
        for r in failures[:5]:
            content = r.get("content", "")[:150]
            print(f"    idx={r['index']}: {content!r}")


def _compute_cell_consistency(records: list[dict]) -> dict:
    cells: dict[str, dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for r in records:
        cat, parsed = _parse_record(r)
        if cat == "day_boundary":
            cell_idx = r["index"] // SAMPLES_DAY_BOUNDARY
            if parsed is not None:
                cells["day_boundary"][cell_idx][parsed] += 1
        elif cat == "within_day":
            cell_idx = (r["index"] - WITHIN_DAY_START) // SAMPLES_WITHIN_DAY
            if parsed is not None:
                _, sb = parsed
                cells["within_day"][cell_idx][sb] += 1
    return dict(cells)


def _print_consistency(label: str, records: list[dict]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}: cell consistency")
    print(f"{'=' * 60}")

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
        print(f"\n  {cat}: {len(cat_cells)} cells, n={n_samples} samples each")
        print(f"    mean agreement: {arr.mean():.1%}")
        print(f"    min agreement:  {arr.min():.1%}")
        print(f"    100% agreement cells: {(arr == 1.0).sum()} / {len(arr)}")
        print(f"    <80% agreement cells: {(arr < 0.8).sum()} / {len(arr)}")


def _compute_day_boundary_transitions(records: list[dict]) -> dict:
    counts: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[tuple, int] = defaultdict(int)
    for r in records:
        cat, parsed = _parse_record(r)
        if cat != "day_boundary" or parsed is None:
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
            if cat != "within_day" or parsed is None:
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
            counts[key][sb] += 1
            totals[key] += 1
        probs = {}
        for key, outcomes in counts.items():
            tt = totals[key]
            probs[key] = {o: c / tt for o, c in outcomes.items()}
        all_probs[t] = probs
    return all_probs


def _print_day_boundary_realism(label: str, records: list[dict]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}: day-boundary transition realism")
    print(f"{'=' * 60}")

    probs = _compute_day_boundary_transitions(records)
    if not probs:
        print("  No parsed day-boundary records.")
        return

    gc = sum(probs[k].get("good", 0) * sum(probs[k].values()) for k in probs)
    tc = sum(sum(probs[k].values()) for k in probs)
    p_good = gc / tc if tc else 0
    print(f"\n  Marginal P(good sleep): {p_good:.1%}")

    for sb in STEP_BINS:
        keys = [k for k in probs if k[0] == sb]
        if not keys:
            continue
        gc = sum(probs[k].get("good", 0) * sum(probs[k].values()) for k in keys)
        tc = sum(sum(probs[k].values()) for k in keys)
        val = gc / tc if tc else 0
        print(f"  P(good | step_bin_daily={sb}): {val:.1%}")

    for b in BURDENS:
        keys = [k for k in probs if k[1] == b]
        if not keys:
            continue
        gc = sum(probs[k].get("good", 0) * sum(probs[k].values()) for k in keys)
        tc = sum(sum(probs[k].values()) for k in keys)
        val = gc / tc if tc else 0
        print(f"  P(good | burden={b}): {val:.1%}")

    for s in SLEEP_TYPES:
        keys = [k for k in probs if k[3] == s]
        if not keys:
            continue
        gc = sum(probs[k].get("good", 0) * sum(probs[k].values()) for k in keys)
        tc = sum(sum(probs[k].values()) for k in keys)
        val = gc / tc if tc else 0
        print(f"  P(good | sleep_last={s}): {val:.1%}")

    for d in DAY_TYPES:
        keys = [k for k in probs if k[2] == d]
        if not keys:
            continue
        gc = sum(probs[k].get("good", 0) * sum(probs[k].values()) for k in keys)
        tc = sum(sum(probs[k].values()) for k in keys)
        val = gc / tc if tc else 0
        print(f"  P(good | day_type={d}): {val:.1%}")


def _print_within_day_realism(label: str, records: list[dict]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}: within-day transition realism")
    print(f"{'=' * 60}")

    all_probs = _compute_within_day_transitions(records)

    for t in range(TIMESTEPS):
        probs = all_probs.get(t, {})
        if not probs:
            continue
        print(f"\n  Timestep {t}:")
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
                print(f"    current={sb}: {parts}")

    print("\n  Action effect (averaged across timesteps):")
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
            print(f"    {action}: {parts}")

    print("\n  Burden effect (averaged across timesteps):")
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
            print(f"    {burden}: {parts}")

    print("\n  Sleep effect (averaged across timesteps):")
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
            print(f"    sleep={sleep}: {parts}")

    print("\n  Monotonicity check P(active | current_bin):")
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
        print(
            f"    timestep {t}: {vals[0]:.1%} <= "
            f"{vals[1]:.1%} <= {vals[2]:.1%}  [{status}]"
        )


def _plot_parse_rate(records_per_model: dict[str, list[dict]], fig_dir: Path) -> None:
    models = list(records_per_model.keys())
    db_rates = []
    wd_rates = []
    for records in records_per_model.values():
        db = [r for r in records if r["index"] < WITHIN_DAY_START]
        wd = [r for r in records if r["index"] >= WITHIN_DAY_START]
        db_p = sum(1 for r in db if _parse_record(r)[1] is not None)
        wd_p = sum(1 for r in wd if _parse_record(r)[1] is not None)
        db_rates.append(db_p / len(db) if db else 0)
        wd_rates.append(wd_p / len(wd) if wd else 0)

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, db_rates, w, label="Day-boundary", color="#3498db")
    ax.bar(x + w / 2, wd_rates, w, label="Within-day", color="#2ecc71")
    ax.set_ylabel("Parse Rate")
    ax.set_title("Parse Rate by Model and Category")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.legend()
    for i, (d, wv) in enumerate(zip(db_rates, wd_rates)):
        ax.text(i - w / 2, d + 0.02, f"{d:.1%}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, wv + 0.02, f"{wv:.1%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "parse_rate.png", dpi=150)
    plt.close(fig)


def _plot_output_dist(records_per_model: dict[str, list[dict]], fig_dir: Path) -> None:
    for label, records in records_per_model.items():
        sleep_dist: dict[str, int] = defaultdict(int)
        sb_dist: dict[str, int] = defaultdict(int)
        for r in records:
            cat, parsed = _parse_record(r)
            if cat == "day_boundary" and parsed is not None:
                sleep_dist[parsed] += 1
            elif cat == "within_day" and parsed is not None:
                _, sb = parsed
                sb_dist[sb] += 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        if sleep_dist:
            keys = ["good", "poor"]
            vals = [sleep_dist.get(k, 0) for k in keys]
            ax1.bar(keys, vals, color=["#2ecc71", "#e74c3c"])
            ax1.set_title(f"{label} — Sleep Quality Distribution")
            ax1.set_ylabel("Count")
            for i, v in enumerate(vals):
                ax1.text(i, v + 20, f"{v:,}", ha="center", va="bottom", fontsize=9)

        if sb_dist:
            keys = ["inactive", "moderate", "active"]
            vals = [sb_dist.get(k, 0) for k in keys]
            colors = [STEP_BIN_COLORS[k] for k in keys]
            ax2.bar(keys, vals, color=colors)
            ax2.set_title(f"{label} — Step Bin Distribution")
            ax2.set_ylabel("Count")
            for i, v in enumerate(vals):
                ax2.text(i, v + 20, f"{v:,}", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        fig.savefig(fig_dir / "output_distribution.png", dpi=150)
        plt.close(fig)


def _plot_cell_consistency(
    records_per_model: dict[str, list[dict]], fig_dir: Path
) -> None:
    for label, records in records_per_model.items():
        cells = _compute_cell_consistency(records)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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
            ax.axvline(
                arr.mean(), color="red", linestyle="--", label=f"mean={arr.mean():.1%}"
            )
            ax.set_xlabel("Agreement Fraction")
            ax.set_ylabel("Number of Cells")
            ax.set_title(f"{label} — {cat.replace('_', ' ').title()} Cell Consistency")
            ax.legend()

        fig.tight_layout()
        fig.savefig(fig_dir / "cell_consistency.png", dpi=150)
        plt.close(fig)


def _plot_step_hist(records_per_model: dict[str, list[dict]], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, records in records_per_model.items():
        steps = []
        for r in records:
            cat, parsed = _parse_record(r)
            if cat == "within_day" and parsed is not None:
                s, _ = parsed
                steps.append(s)
        if steps:
            ax.hist(steps, bins=50, alpha=0.5, label=label, edgecolor="white")
    ax.set_xlabel("Step Count")
    ax.set_ylabel("Frequency")
    ax.set_title("Step Count Distribution (Within-Day)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "step_histogram.png", dpi=150)
    plt.close(fig)


def _plot_day_boundary_heatmap(
    records_per_model: dict[str, list[dict]], fig_dir: Path
) -> None:
    for label, records in records_per_model.items():
        probs = _compute_day_boundary_transitions(records)
        if not probs:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        for ax, burden in zip(axes, BURDENS):
            data = np.zeros((len(STEP_BINS), len(DAY_TYPES)))
            for i, sb in enumerate(STEP_BINS):
                for j, d in enumerate(DAY_TYPES):
                    key = (sb, burden, d, "good")
                    data[i, j] = probs.get(key, {}).get("good", 0.5)
            im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(DAY_TYPES)))
            ax.set_xticklabels(DAY_TYPES)
            ax.set_yticks(range(len(STEP_BINS)))
            ax.set_yticklabels(STEP_BINS)
            ax.set_title(f"burden={burden}")
            for i in range(len(STEP_BINS)):
                for j in range(len(DAY_TYPES)):
                    ax.text(
                        j, i, f"{data[i, j]:.1%}", ha="center", va="center", fontsize=9
                    )

        fig.suptitle(
            f"{label} — P(good sleep | step_bin_daily, burden, day_type, sleep=good)",
            y=1.02,
        )
        fig.colorbar(im, ax=axes, shrink=0.8, label="P(good)")
        fig.tight_layout()
        fig.savefig(fig_dir / "day_boundary_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_within_day_action(
    records_per_model: dict[str, list[dict]], fig_dir: Path
) -> None:
    for label, records in records_per_model.items():
        all_probs = _compute_within_day_transitions(records)
        fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
        for t, ax in enumerate(axes):
            probs = all_probs.get(t, {})
            x = np.arange(len(ACTIONS))
            width = 0.25
            for i, sb in enumerate(STEP_BINS):
                vals = []
                for action in ACTIONS:
                    keys = [k for k in probs if k[0] == "moderate" and k[2] == action]
                    if keys:
                        dist: dict[str, float] = defaultdict(float)
                        total = 0
                        for k in keys:
                            w = sum(probs[k].values())
                            for o, p in probs[k].items():
                                dist[o] += p * w
                            total += w
                        vals.append(dist.get(sb, 0) / total if total else 0)
                    else:
                        vals.append(0)
                ax.bar(
                    x + i * width,
                    vals,
                    width,
                    label=sb,
                    color=STEP_BIN_COLORS[sb],
                    alpha=0.8,
                )
            ax.set_xticks(x + width)
            ax.set_xticklabels([a[:4] for a in ACTIONS], fontsize=8)
            ax.set_title(f"t={t}")
            if t == 0:
                ax.set_ylabel("Probability")
                ax.legend(fontsize=8)

        fig.suptitle(f"{label} — P(step_bin' | action) for current=moderate", y=1.02)
        fig.tight_layout()
        fig.savefig(fig_dir / "within_day_action.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_burden_interaction(
    records_per_model: dict[str, list[dict]], fig_dir: Path
) -> None:
    for label, records in records_per_model.items():
        all_probs = _compute_within_day_transitions(records)
        fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

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
            ax.set_xlabel("Burden Level")
            ax.set_title(f"action={action}")
            if action == ACTIONS[0]:
                ax.set_ylabel("P(active next)")
                ax.legend(fontsize=8)

        fig.suptitle(f"{label} — P(active | burden) by step_bin and action", y=1.02)
        fig.tight_layout()
        fig.savefig(fig_dir / "burden_interaction.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


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

    logging.basicConfig(level=logging.WARNING)

    model_data: dict[str, list[dict]] = {}
    for fpath in args.files:
        path = Path(fpath)
        label = _model_label(path)
        print(f"Loading {path} ({label})...")
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

        print(f"\n  Figures saved to {out_dir}/")

    print(f"\n{'=' * 60}")
    print("  Done.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
