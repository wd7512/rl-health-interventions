#!/usr/bin/env python3
"""Generate random PEARL transition tables for the 12-action COM-B action space.

Outputs a single JSON file containing all 108 states by 13 actions = 1,404
transition records with randomly generated probability distributions for each
stochastic factor using numpy's Dirichlet distribution.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── PEARL 12-action config ───────────────────────────────────────────────────

STOCHASTIC_FACTORS: dict[str, list[str]] = {
    "recent_steps_mean": ["low", "moderate", "high"],
    "recent_walk_pattern": ["low", "high"],
    "morning_steps_ratio": ["morning", "balanced", "evening"],
}

DETERMINISTIC_FACTORS: dict[str, list[str]] = {
    "day_of_week": ["weekday", "weekend"],
    "burden": ["low", "medium", "high"],
}

COM_B_THEMES: list[str] = [
    "ability",
    "perceived_benefit",
    "planning",
    "prioritization",
    "social_opportunity",
    "physical_opportunity",
]

DELIVERY_TIMES: list[str] = ["morning", "afternoon"]

ACTIONS: list[str] = ["idle"] + [
    f"{theme}_{delivery}" for theme in COM_B_THEMES for delivery in DELIVERY_TIMES
]

# Derived cardinalities
N_STOCHASTIC_LEVELS: list[int] = [len(v) for v in STOCHASTIC_FACTORS.values()]
N_STATES: int = (
    len(DETERMINISTIC_FACTORS["day_of_week"])
    * len(DETERMINISTIC_FACTORS["burden"])
    * N_STOCHASTIC_LEVELS[0]
    * N_STOCHASTIC_LEVELS[1]
    * N_STOCHASTIC_LEVELS[2]
)
N_ACTIONS: int = len(ACTIONS)
N_RECORDS: int = N_STATES * N_ACTIONS

OUTPUT_FILENAME: str = "pearl_random.json"

# Tolerance for floating-point sum checks
_PROB_SUM_TOLERANCE: float = 1e-6


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_all_states() -> list[dict[str, str]]:
    """Build the Cartesian product of all 5 state factors.

    Returns
    -------
    list[dict[str, str]]
        List of all 108 state definitions in deterministic order.
    """
    factor_names = [
        "recent_steps_mean",
        "recent_walk_pattern",
        "morning_steps_ratio",
        "day_of_week",
        "burden",
    ]
    factor_values = [
        STOCHASTIC_FACTORS["recent_steps_mean"],
        STOCHASTIC_FACTORS["recent_walk_pattern"],
        STOCHASTIC_FACTORS["morning_steps_ratio"],
        DETERMINISTIC_FACTORS["day_of_week"],
        DETERMINISTIC_FACTORS["burden"],
    ]
    return [
        dict(zip(factor_names, combo, strict=True))
        for combo in itertools.product(*factor_values)
    ]


def _random_prob_map(
    rng: np.random.Generator,
    levels: list[str],
) -> dict[str, float]:
    """Generate a random probability distribution over ``levels``.

    Uses Dirichlet(alpha=1) and renormalises after rounding to 4 decimal
    places so the result sums exactly to 1.0.

    Parameters
    ----------
    rng : numpy.random.Generator
        NumPy random generator.
    levels : list[str]
        Outcome level names.

    Returns
    -------
    dict[str, float]
        Mapping from level to probability, summing to 1.0.
    """
    probs = rng.dirichlet(np.ones(len(levels), dtype=np.float64))
    prob_map = {
        level: round(float(prob), 4) for level, prob in zip(levels, probs, strict=True)
    }
    # Renormalise after rounding
    total = sum(prob_map.values())
    if total > 0:
        diff = round(1.0 - total, 4)
        max_key = max(prob_map, key=prob_map.__getitem__)  # type: ignore[arg-type]
        prob_map[max_key] = round(prob_map[max_key] + diff, 4)
    return prob_map


def generate_transitions(seed: int) -> list[dict[str, Any]]:
    """Generate all 1,404 transition records with random probability vectors.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[dict[str, Any]]
        List of transition records, each with ``state``, ``action``,
        and ``next_state_probs`` keys.
    """
    rng = np.random.default_rng(seed)
    states = _build_all_states()

    assert len(states) == N_STATES, f"Expected {N_STATES} states, got {len(states)}"

    transitions: list[dict[str, Any]] = []
    for state in states:
        for action in ACTIONS:
            next_state_probs = {
                factor: _random_prob_map(rng, levels)
                for factor, levels in STOCHASTIC_FACTORS.items()
            }
            transitions.append(
                {
                    "state": state,
                    "action": action,
                    "next_state_probs": next_state_probs,
                }
            )

    assert len(transitions) == N_RECORDS, (
        f"Expected {N_RECORDS} transitions, got {len(transitions)}"
    )
    return transitions


def _validate_entry_state(
    entry: dict[str, Any],
    record_tag: str,
    errors: list[str],
) -> None:
    """Check that the entry has all expected state factors."""
    all_state_keys = set(STOCHASTIC_FACTORS) | set(DETERMINISTIC_FACTORS)
    state_keys = set(entry.get("state", {}).keys())
    missing_state = all_state_keys - state_keys
    if missing_state:
        errors.append(f"{record_tag}: missing state factors: {sorted(missing_state)}")


def _validate_entry_nsp_keys(
    nsp: dict[str, Any],
    record_tag: str,
    errors: list[str],
) -> None:
    """Check that next_state_probs has all stochastic factors."""
    nsp_keys = set(nsp.keys())
    missing_stochastic = set(STOCHASTIC_FACTORS) - nsp_keys
    if missing_stochastic:
        errors.append(
            f"{record_tag}: missing stochastic factors in "
            f"next_state_probs: {sorted(missing_stochastic)}"
        )


def _check_level_coverage(
    factor: str,
    levels: list[str],
    dist: dict[str, float],
    record_tag: str,
    errors: list[str],
) -> None:
    """Check that all outcome levels appear in the distribution."""
    for level in levels:
        if level not in dist:
            errors.append(
                f"{record_tag}.next_state_probs.{factor}: missing level {level!r}"
            )


def _check_non_negative(
    factor: str,
    dist: dict[str, float],
    record_tag: str,
    errors: list[str],
) -> None:
    """Check that no probability is negative."""
    for level_val, prob in dist.items():
        if prob < 0:
            errors.append(
                f"{record_tag}.next_state_probs.{factor}.{level_val}: "
                f"negative probability {prob}"
            )


def _check_sum_to_one(
    factor: str,
    dist: dict[str, float],
    record_tag: str,
    errors: list[str],
) -> None:
    """Check that distribution sums to 1.0 within tolerance."""
    total = sum(dist.values())
    if abs(total - 1.0) > _PROB_SUM_TOLERANCE:
        errors.append(
            f"{record_tag}.next_state_probs.{factor}: "
            f"probabilities sum to {total}, expected 1.0"
        )


def _validate_distribution(
    factor: str,
    levels: list[str],
    dist: dict[str, float],
    record_tag: str,
    errors: list[str],
) -> None:
    """Validate a single probability distribution.

    Delegates to individual check functions for levels coverage,
    non-negativity, and sum to 1.0.
    """
    _check_level_coverage(factor, levels, dist, record_tag, errors)
    _check_non_negative(factor, dist, record_tag, errors)
    _check_sum_to_one(factor, dist, record_tag, errors)


def validate_transitions(transitions: list[dict[str, Any]]) -> list[str]:
    """Validate transition records and return a list of error messages.

    Checks:
    - All 3 stochastic factors present in every record
    - Each probability distribution sums to 1.0
    - All probability values are non-negative
    - All 5 state factors present

    Parameters
    ----------
    transitions : list[dict[str, Any]]
        List of transition records.

    Returns
    -------
    list[str]
        List of error messages. Empty means all checks passed.
    """
    errors: list[str] = []

    for idx, entry in enumerate(transitions):
        record_tag = f"record[{idx}]"
        _validate_entry_state(entry, record_tag, errors)

        nsp = entry.get("next_state_probs", {})
        _validate_entry_nsp_keys(nsp, record_tag, errors)

        for factor, levels in STOCHASTIC_FACTORS.items():
            dist = nsp.get(factor, {})
            _validate_distribution(factor, levels, dist, record_tag, errors)

    return errors


def build_output(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the full output dictionary with ``global_state`` and ``transitions``.

    Parameters
    ----------
    transitions : list[dict[str, Any]]
        List of transition records.

    Returns
    -------
    dict[str, Any]
        Output dictionary ready for JSON serialisation.
    """
    return {
        "global_state": {"step_of_day": 0},
        "transitions": transitions,
    }


def write_output(
    output: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Write the output dictionary to a JSON file.

    Parameters
    ----------
    output : dict[str, Any]
        Output dictionary.
    output_dir : Path
        Output directory.

    Returns
    -------
    Path
        Path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info("Wrote %s", output_path)
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate random PEARL transition tables for the "
        "12-action COM-B action space.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tables/pearl_12action",
        help=(
            "Output directory for the generated table (default: tables/pearl_12action)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the table generation script.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for errors).
    """
    args = parse_args(argv)

    output_dir = Path(args.output_dir)
    seed = args.seed

    logger.info(
        "Generating PEARL 12-action random transition table with seed=%d",
        seed,
    )
    logger.info("States: %d, Actions: %d, Records: %d", N_STATES, N_ACTIONS, N_RECORDS)

    # Generate
    transitions = generate_transitions(seed)

    # Validate
    errors = validate_transitions(transitions)
    if errors:
        for err in errors:
            logger.error("Validation error: %s", err)
        logger.error("Validation failed with %d error(s)", len(errors))
        return 1

    logger.info("Validation passed: all %d records valid", len(transitions))

    # Build output
    output = build_output(transitions)

    # Write
    output_path = write_output(output, output_dir)
    logger.info("Done — output: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
