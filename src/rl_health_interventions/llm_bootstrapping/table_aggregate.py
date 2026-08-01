"""Aggregate LLM bootstrap responses into a PEARL transition table.

Shared by the mini and full-scale generator scripts. Turns per-prompt
LLM responses (parsed 7-day histories) into a transition table with
per-factor next-state probability distributions.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict

from rl_health_interventions.llm_bootstrapping.parse_pearl import (
    history_to_factors,
    parse_day_history,
)

logger = logging.getLogger(__name__)

MIN_SAMPLES_PER_CELL = 2

# The three stochastic factors aggregated per transition cell.
AGGREGATED_FACTORS = [
    "recent_steps_mean",
    "recent_walk_pattern",
    "morning_steps_ratio",
]


def _normalise_probs(counts: dict[str, int]) -> dict[str, float]:
    """Convert factor-value counts into probabilities that sum to 1.0.

    Naive ``round(v / total, 4)`` on every value can leave the sum at
    0.9999 (e.g. 1/1/1 at n=3, 2/2/2 at n=6), which trips the
    ``TableValidator`` probability-sum check (|sum - 1.0| <= 1e-6). To
    avoid this, every value except the last is rounded and the last value
    absorbs the rounding remainder (last = 1 - sum(others)).
    """
    if not counts:
        return {}
    total = sum(counts.values())
    items = sorted(counts.items())
    probs: dict[str, float] = {}
    for value, count in items[:-1]:
        probs[value] = round(count / total, 4)
    remainder = round(1.0 - sum(probs.values()), 4)
    probs[items[-1][0]] = max(0.0, remainder)
    return probs


def aggregate_to_table(  # noqa: C901, PLR0912
    results: list[dict],
    state_action_pairs: list[tuple[dict, str]],
    *,
    min_samples_per_cell: int = MIN_SAMPLES_PER_CELL,
) -> dict:
    """Aggregate LLM responses into a transition table.

    Parameters
    ----------
    results : list[dict]
        LLM batch results with 'content' or 'error' keys.
    state_action_pairs : list[tuple[dict, str]]
        Corresponding (state, action) for each prompt.
    min_samples_per_cell : int
        Cells with fewer parsed samples than this are dropped.

    Returns
    -------
    dict in pearl_random.json format: {"global_state": {}, "transitions": [...]}.
    """
    cell_results: dict[str, list[dict[str, str]]] = defaultdict(list)
    state_lookup: dict[str, dict] = {}

    for result, (state, action) in zip(results, state_action_pairs, strict=True):
        if "error" in result:
            continue

        content = result.get("content", "")
        history = parse_day_history(content)
        if history is None:
            continue

        factors = history_to_factors(history)
        state_key = json.dumps(state, sort_keys=True)
        cell_key = f"{state_key}||{action}"
        cell_results[cell_key].append(factors)
        state_lookup[cell_key] = state

    transitions = []
    for cell_key, factor_samples in cell_results.items():
        state = state_lookup[cell_key]
        state_key_str, action = cell_key.rsplit("||", 1)

        if len(factor_samples) < min_samples_per_cell:
            logger.warning(
                "Too few samples for %s/%s: %d",
                state_key_str[:50],
                action,
                len(factor_samples),
            )
            continue

        next_state_probs: dict[str, dict[str, float]] = {}
        for factor in AGGREGATED_FACTORS:
            counts: dict[str, int] = defaultdict(int)
            for sample in factor_samples:
                counts[sample[factor]] += 1
            next_state_probs[factor] = _normalise_probs(dict(counts))

        transitions.append(
            {
                "state": state,
                "action": action,
                "next_state_probs": next_state_probs,
                "n_samples": len(factor_samples),
            }
        )

    return {
        "global_state": {},
        "transitions": transitions,
    }
