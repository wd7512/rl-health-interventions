"""Analyze a PEARL pilot transition table against constitution proxy checks.

Reads an aggregated pilot table (pearl_pilot.json format) and computes the
metric set used by the prompt-refinement log
(docs/research/prompt-refinement-log.md / .json). Purely local analysis —
no LLM calls — so it is safe to run after every generation round.

Checks (constitution-aligned proxies, thresholds in CHECK_THRESHOLDS):
  C1 action coverage: every PEARL action appears in the table.
  C2 cell coverage: every (state, action) cell from the present states is filled.
  C3 state persistence: idle keeps a 'high' recent_steps_mean state high
     (and a 'low' state low).
  C4 action sensitivity: ability_morning raises P(high) vs idle in a majority
     of cells.
  C5 burden monotonicity: for each recent_steps_mean level, mean P(high) under
     major burden does not exceed mean P(high) under no burden.
  C6 factor variation: morning_steps_ratio and recent_walk_pattern are not
     collapsed onto a single value across the table.

Usage:
    uv run python scripts/pearl_recalibration/analyze_pearl_mini.py
        [--table PATH] [--json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl_health_interventions.llm_bootstrapping._shared import (  # noqa: E402
    setup_logging,
)
from rl_health_interventions.llm_bootstrapping.prompts.pearl import (  # noqa: E402
    ACTIONS,
)

logger = logging.getLogger(__name__)

TIME_VARYING_FACTORS = [
    "recent_steps_mean",
    "recent_walk_pattern",
    "morning_steps_ratio",
]
MONOTONICITY_FACTOR = "recent_steps_mean"
INTERVENTION_ACTION = "ability_morning"
CONTROL_ACTION = "idle"
_MIN_TRIMMED_LIFTS = 4

CHECK_THRESHOLDS: dict[str, float] = {
    "min_action_coverage": 1.0,  # all actions present (fraction)
    "min_cell_coverage": 1.0,  # all expected (state, action) cells filled
    "min_persistence_high": 0.5,  # idle P(high) in a high state
    "min_persistence_low": 0.5,  # idle P(low) in a low state
    "min_sensitivity_frac": 0.5,  # fraction of cells with dP(high) > 0
    "max_factor_dominant_share": 0.75,  # no single value in >75% of cells
}


def _load_table(table_path: Path) -> dict:
    with table_path.open() as f:
        return json.load(f)


def _load_raw_results(raw_path: Path) -> list[dict]:
    """Load the raw LLM results jsonl (state, action, content|error)."""
    records = []
    with raw_path.open() as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _mean_daily_steps(content: str) -> float | None:
    """Parse a 7-day history and return the mean daily steps, or None."""
    from rl_health_interventions.llm_bootstrapping.parse_pearl import (  # noqa: PLC0415
        parse_day_history,
    )

    history = parse_day_history(content)
    if not history:
        return None
    totals = [d["morning_steps"] + d["afternoon_steps"] for d in history]
    return sum(totals) / len(totals)


def compute_raw_effect(  # noqa: C901, PLR0912, PLR0915
    raw_records: list[dict],
) -> dict:
    """Compute intervention-vs-idle step lift directly from raw LLM output.

    The bin-based C4 check is structurally blind in the pilot subset (idle
    persists at P(high)=1.0 in high states; low states cannot cross the
    >7,000 bin), so this measures the actual step response instead.
    """
    # (state_key, action) -> list of mean daily steps
    cell_means: dict[tuple[str, str], list[float]] = defaultdict(list)
    state_lookup: dict[tuple[str, str], dict] = {}
    n_parsed = 0
    for record in raw_records:
        if "error" in record:
            continue
        mean = _mean_daily_steps(record.get("content", ""))
        if mean is None:
            continue
        state_key = json.dumps(record["state"], sort_keys=True)
        cell_means[(state_key, record["action"])].append(mean)
        state_lookup[(state_key, record["action"])] = record["state"]
        n_parsed += 1

    per_state: list[dict] = []
    by_state: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for (state_key, action), means in cell_means.items():
        by_state[state_key][action] = means

    n_lift_cells = 0
    n_state_cells = 0
    all_lifts: list[float] = []
    for state_key, action_means in sorted(by_state.items()):
        if CONTROL_ACTION not in action_means:
            continue
        idle_mean = sum(action_means[CONTROL_ACTION]) / len(
            action_means[CONTROL_ACTION]
        )
        state_cells = 0
        state_lifts = 0
        for action, means in sorted(action_means.items()):
            if action == CONTROL_ACTION:
                continue
            action_mean = sum(means) / len(means)
            lift = action_mean - idle_mean
            state_lifts += lift
            state_cells += 1
            all_lifts.append(lift)
            if lift > 0:
                n_lift_cells += 1
        n_state_cells += state_cells
        per_state.append(
            {
                "state": state_lookup[(state_key, CONTROL_ACTION)],
                "idle_mean_steps": round(idle_mean, 1),
                "n_intervention_cells": state_cells,
                "mean_lift_steps": round(state_lifts / state_cells, 1)
                if state_cells
                else None,
            }
        )

    # Robust summaries (Option B, literature-backed): the mean is the
    # round-comparison metric (back-compatible with rounds 1-10); median
    # and trimmed mean quantify outlier sensitivity (e.g. the round-10
    # -614.3 cell) without changing the logged metric.
    sorted_lifts = sorted(all_lifts)
    n_lifts = len(sorted_lifts)
    median_lift: float | None = None
    if n_lifts:
        if n_lifts % 2 == 1:
            median_lift = round(sorted_lifts[n_lifts // 2], 1)
        else:
            median_lift = round(
                (sorted_lifts[n_lifts // 2 - 1] + sorted_lifts[n_lifts // 2]) / 2, 1
            )

    def _trimmed_mean(values: list[float]) -> float | None:
        if len(values) < _MIN_TRIMMED_LIFTS:
            return None
        return round(sum(values[1:-1]) / (len(values) - 2), 1)

    return {
        "n_records": len(raw_records),
        "n_parsed": n_parsed,
        "per_state": per_state,
        "n_lift_cells": n_lift_cells,
        "n_state_cells": n_state_cells,
        "mean_lift_steps": round(sum(all_lifts) / len(all_lifts), 1)
        if all_lifts
        else None,
        "median_lift_steps": median_lift if all_lifts else None,
        "trimmed_mean_lift_steps": _trimmed_mean(sorted_lifts),
        "min_lift_steps": round(sorted_lifts[0], 1) if all_lifts else None,
        "max_lift_steps": round(sorted_lifts[-1], 1) if all_lifts else None,
    }


def _per_cell(transitions: list[dict]) -> dict[tuple[str, str], dict]:
    """Group transitions by (state_key, action) and return next_state_probs."""
    cells: dict[tuple[str, str], dict] = {}
    for t in transitions:
        state_key = json.dumps(t["state"], sort_keys=True)
        cells[(state_key, t["action"])] = t["next_state_probs"]
    return cells


def compute_metrics(table: dict) -> dict:  # noqa: C901, PLR0912, PLR0915
    """Compute the full metric set for a pilot table."""
    transitions = table.get("transitions", [])
    cells = _per_cell(transitions)

    states: list[dict] = []
    state_actions: dict[str, set[str]] = defaultdict(set)
    for t in transitions:
        key = json.dumps(t["state"], sort_keys=True)
        if key not in state_actions:
            states.append(t["state"])
        state_actions[key].add(t["action"])

    n_actions_present = len({a for _k, a in cells})
    n_expected_cells = len(states) * len(ACTIONS)
    n_cells = len(cells)

    # C6: factor variation across cells — the modal value of each cell's
    # distribution must not be the same value in too many cells.
    modal_value_counts: dict[str, Counter] = {
        factor: Counter() for factor in TIME_VARYING_FACTORS
    }
    for next_probs in cells.values():
        for factor in TIME_VARYING_FACTORS:
            probs = next_probs.get(factor, {})
            best_value, best_prob = None, -1.0
            for value, prob in probs.items():
                if prob > best_prob:
                    best_value, best_prob = value, prob
            if best_value is not None:
                modal_value_counts[factor][best_value] += 1

    # C3: persistence on idle
    persistence: dict[str, dict[str, float | None]] = {}
    for level in ("low", "high"):
        idle_ps = [
            next_probs.get("recent_steps_mean", {}).get(level, 0.0)
            for (state_key, action), next_probs in cells.items()
            if action == CONTROL_ACTION
            and json.loads(state_key)["recent_steps_mean"] == level
        ]
        persistence[level] = {
            "n_cells": len(idle_ps),
            "mean_p_stay": round(sum(idle_ps) / len(idle_ps), 4) if idle_ps else None,
        }

    # C4: intervention sensitivity vs idle
    sensitivity: list[dict] = []
    for (state_key, action), next_probs in cells.items():
        if action != INTERVENTION_ACTION:
            continue
        idle_probs = cells.get((state_key, CONTROL_ACTION))
        if idle_probs is None:
            continue
        d_high = next_probs.get("recent_steps_mean", {}).get(
            "high", 0.0
        ) - idle_probs.get("recent_steps_mean", {}).get("high", 0.0)
        sensitivity.append(
            {
                "state": json.loads(state_key),
                "dP(high)": round(d_high, 4),
            }
        )
    n_sensitive = sum(1 for s in sensitivity if s["dP(high)"] > 0)

    # C5: burden monotonicity, mean P(high) per (rsm_level, burden)
    burden_means: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (state_key, _action), next_probs in cells.items():
        state = json.loads(state_key)
        p_high = next_probs.get("recent_steps_mean", {}).get("high", 0.0)
        burden_means[(state["recent_steps_mean"], state["burden"])].append(p_high)
    monotonicity = {}
    for rsm_level in ("low", "high"):
        none_ps = burden_means.get((rsm_level, "none"), [])
        major_ps = burden_means.get((rsm_level, "major"), [])
        if none_ps and major_ps:
            mean_none = sum(none_ps) / len(none_ps)
            mean_major = sum(major_ps) / len(major_ps)
            monotonicity[rsm_level] = {
                "mean_P(high)_none": round(mean_none, 4),
                "mean_P(high)_major": round(mean_major, 4),
                "burden_reduces_steps": mean_major <= mean_none,
            }

    # Verdicts
    th = CHECK_THRESHOLDS
    action_coverage = round(n_actions_present / len(ACTIONS), 4)
    cell_coverage = round(n_cells / n_expected_cells, 4) if n_expected_cells else 1.0
    sens_frac = round(n_sensitive / len(sensitivity), 4) if sensitivity else 0.0

    dominant_share_fracs: dict[str, float] = {}
    for factor in TIME_VARYING_FACTORS:
        counts = modal_value_counts[factor]
        dominant_share_fracs[factor] = (
            round(max(counts.values()) / n_cells, 4) if n_cells else 0.0
        )
    factor_variation = {
        factor: {
            "n_distinct_values": len(modal_value_counts[factor]),
            "dominant_value": modal_value_counts[factor].most_common(1)[0][0]
            if modal_value_counts[factor]
            else None,
            "dominant_share": share,
        }
        for factor, share in dominant_share_fracs.items()
    }

    checks = {
        "C1_action_coverage": {
            "pass": action_coverage >= th["min_action_coverage"],
            "actual": action_coverage,
            "threshold": th["min_action_coverage"],
            "detail": f"{n_actions_present}/{len(ACTIONS)} actions present",
        },
        "C2_cell_coverage": {
            "pass": cell_coverage >= th["min_cell_coverage"],
            "actual": cell_coverage,
            "threshold": th["min_cell_coverage"],
            "detail": f"{n_cells}/{n_expected_cells} cells filled",
        },
        "C3_state_persistence": {
            "pass": (
                persistence["low"]["mean_p_stay"] is not None
                and persistence["high"]["mean_p_stay"] is not None
                and persistence["low"]["mean_p_stay"] >= th["min_persistence_low"]
                and persistence["high"]["mean_p_stay"] >= th["min_persistence_high"]
            ),
            "actual": {
                "low": persistence["low"]["mean_p_stay"],
                "high": persistence["high"]["mean_p_stay"],
            },
            "threshold": {
                "low": th["min_persistence_low"],
                "high": th["min_persistence_high"],
            },
            "detail": "idle keeps low states low and high states high",
        },
        "C4_action_sensitivity": {
            "pass": sens_frac >= th["min_sensitivity_frac"],
            "actual": sens_frac,
            "threshold": th["min_sensitivity_frac"],
            "detail": (
                f"{n_sensitive}/{len(sensitivity)} cells where ability_morning "
                "raises P(high) vs idle"
            ),
        },
        "C5_burden_monotonicity": {
            "pass": all(m["burden_reduces_steps"] for m in monotonicity.values())
            and len(monotonicity) > 0,
            "actual": monotonicity,
            "threshold": "major <= none for every recent_steps_mean level",
            "detail": "major burden does not raise mean P(high) vs no burden",
        },
        "C6_factor_variation": {
            "pass": all(
                share <= th["max_factor_dominant_share"]
                for share in dominant_share_fracs.values()
            ),
            "actual": factor_variation,
            "threshold": th["max_factor_dominant_share"],
            "detail": "no factor collapses onto one value",
        },
    }

    return {
        "structure": {
            "n_states": len(states),
            "n_actions": len(ACTIONS),
            "n_actions_present": n_actions_present,
            "n_expected_cells": n_expected_cells,
            "n_cells": n_cells,
            "action_coverage_frac": action_coverage,
            "cell_coverage_frac": cell_coverage,
        },
        "checks": checks,
        "sensitivity": sensitivity,
        "persistence": persistence,
        "monotonicity": monotonicity,
        "factor_variation": factor_variation,
        "modal_value_counts": {
            factor: dict(counts) for factor, counts in modal_value_counts.items()
        },
    }


def main() -> None:  # noqa: C901
    """Print metrics for a pilot table (optionally as JSON)."""
    setup_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        type=Path,
        default=_REPO_ROOT / "tables" / "pearl_12action_pilot" / "pearl_pilot.json",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Raw results jsonl to compute intervention step-lift from",
    )
    parser.add_argument("--json", action="store_true", help="Emit a single JSON object")
    args = parser.parse_args()

    table = _load_table(args.table)
    metrics = compute_metrics(table)

    if args.raw is not None:
        metrics["raw_effect"] = compute_raw_effect(_load_raw_results(args.raw))

    if args.json:
        # Machine-readable payload on stdout (log lines go to stderr via
        # setup_logging), so `--json` output stays pipable.
        print(json.dumps(metrics, indent=2))
        return

    for check_id, check in metrics["checks"].items():
        marker = "PASS" if check["pass"] else "FAIL"
        logger.info("%s: [%s] %s", check_id, marker, check["detail"])
    for level, m in metrics["persistence"].items():
        logger.info("persistence[%s]: mean P(stay) = %s", level, m["mean_p_stay"])
    for rsm_level, m in metrics["monotonicity"].items():
        logger.info(
            "monotonicity[%s]: none=%s major=%s reduces=%s",
            rsm_level,
            m["mean_P(high)_none"],
            m["mean_P(high)_major"],
            m["burden_reduces_steps"],
        )


if __name__ == "__main__":
    main()
