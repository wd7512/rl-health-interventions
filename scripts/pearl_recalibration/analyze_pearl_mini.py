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
            next_probs["recent_steps_mean"].get(level, 0.0)
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
        d_high = next_probs["recent_steps_mean"].get("high", 0.0) - idle_probs[
            "recent_steps_mean"
        ].get("high", 0.0)
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
        p_high = next_probs["recent_steps_mean"].get("high", 0.0)
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


def main() -> None:
    """Print metrics for a pilot table (optionally as JSON)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        type=Path,
        default=_REPO_ROOT / "tables" / "pearl_12action_pilot" / "pearl_pilot.json",
    )
    parser.add_argument("--json", action="store_true", help="Emit a single JSON object")
    args = parser.parse_args()

    table = _load_table(args.table)
    metrics = compute_metrics(table)

    if args.json:
        print(json.dumps(metrics, indent=2))
        return

    for check_id, check in metrics["checks"].items():
        marker = "PASS" if check["pass"] else "FAIL"
        print(f"{check_id}: [{marker}] {check['detail']}")
    for level, m in metrics["persistence"].items():
        print(f"persistence[{level}]: mean P(stay) = {m['mean_p_stay']}")
    for rsm_level, m in metrics["monotonicity"].items():
        print(
            f"monotonicity[{rsm_level}]: none={m['mean_P(high)_none']} "
            f"major={m['mean_P(high)_major']} reduces={m['burden_reduces_steps']}"
        )


if __name__ == "__main__":
    main()
