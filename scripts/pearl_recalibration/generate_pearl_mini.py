"""Generate a mini PEARL transition table using LLM bootstrapping.

Pilot script: generates a small table (~4 states x 13 actions = 52 cells)
to validate the pipeline end-to-end before scaling to full 108-state table.

States include all 5 MDP factors (recent_steps_mean, recent_walk_pattern,
morning_steps_ratio, day_of_week, burden).
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl_health_interventions.llm_bootstrapping._shared import (  # noqa: E402
    load_env,
    setup_logging,
)
from rl_health_interventions.llm_bootstrapping.parse_pearl import (  # noqa: E402
    history_to_factors,
    parse_day_history,
)
from rl_health_interventions.llm_bootstrapping.prompts.pearl import (  # noqa: E402
    ACTIONS,
    generate_prompts,
)
from rl_health_interventions.llm_bootstrapping.request import (  # noqa: E402
    batch_complete,
)

logger = logging.getLogger(__name__)

# Mini-table: subset of the full 108-state space.
# For the pilot, use 2 burden levels x 2 recent_steps_mean levels,
# keeping other factors fixed = 4 states x 13 actions = 52 cells.
MINI_STATES = [
    {
        "recent_steps_mean": rsm,
        "recent_walk_pattern": "low",
        "morning_steps_ratio": "balanced",
        "day_of_week": "weekday",
        "burden": b,
    }
    for rsm in ("low", "high")
    for b in ("none", "major")
]

SAMPLES_PER_CELL = 5


_MIN_SAMPLES_PER_CELL = 2


def _aggregate_to_table(  # noqa: C901, PLR0912
    results: list[dict],
    state_action_pairs: list[tuple[dict, str]],
) -> dict:
    """Aggregate LLM responses into a transition table.

    Parameters
    ----------
    results : list[dict]
        LLM batch results with 'content' or 'error' keys.
    state_action_pairs : list[tuple[dict, str]]
        Corresponding (state, action) for each prompt.

    Returns
    -------
    dict in pearl_random.json format.
    """
    # Group results by (state_key, action)
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

    # Build transition table
    transitions = []
    for cell_key, factor_samples in cell_results.items():
        state = state_lookup[cell_key]
        state_key_str, action = cell_key.rsplit("||", 1)

        if len(factor_samples) < _MIN_SAMPLES_PER_CELL:
            logger.warning(
                "Too few samples for %s/%s: %d",
                state_key_str[:50],
                action,
                len(factor_samples),
            )
            continue

        # Count occurrences of each factor value
        next_state_probs = {}
        for factor in [
            "recent_steps_mean",
            "recent_walk_pattern",
            "morning_steps_ratio",
        ]:
            counts: dict[str, int] = defaultdict(int)
            for sample in factor_samples:
                counts[sample[factor]] += 1

            total = len(factor_samples)
            probs = {k: round(v / total, 4) for k, v in counts.items()}
            next_state_probs[factor] = probs

        transitions.append(
            {
                "state": state,
                "action": action,
                "next_state_probs": next_state_probs,
            }
        )

    return {
        "global_state": {},
        "transitions": transitions,
    }


def main() -> None:
    setup_logging()
    load_env()

    samples = int(sys.argv[1]) if len(sys.argv) > 1 else SAMPLES_PER_CELL
    out_dir = Path(_REPO_ROOT / "tables" / "pearl_12action_pilot")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pearl_pilot.json"

    logger.info(
        "Generating mini-table: %d states x %d actions x %d samples",
        len(MINI_STATES),
        len(ACTIONS),
        samples,
    )

    system_prompt, prompts = generate_prompts(
        persona="base",
        samples_per_cell=samples,
        state_subset=MINI_STATES,
    )
    logger.info("Generated %d prompts", len(prompts))

    # Build state-action pairs for aggregation
    state_action_pairs = [
        (state, action)
        for state in MINI_STATES
        for action in ACTIONS
        for _ in range(samples)
    ]

    # Call LLM
    logger.info("Calling LLM...")
    results = batch_complete(
        prompts,
        system_prompt=system_prompt,
        max_workers=50,
        provider="openrouter",
    )

    # Count successes
    ok = sum(1 for r in results if "content" in r)
    logger.info("LLM results: %d/%d succeeded", ok, len(results))

    # Aggregate
    table = _aggregate_to_table(results, state_action_pairs)
    logger.info("Table has %d transitions", len(table["transitions"]))

    # Save
    with out_path.open("w") as f:
        json.dump(table, f, indent=2)
    logger.info("Saved table to %s", out_path)

    # Print summary
    for t in table["transitions"]:
        state = t["state"]
        action = t["action"]
        probs = t["next_state_probs"]["recent_steps_mean"]
        high_prob = probs.get("high", 0)
        low_prob = probs.get("low", 0)
        logger.info(
            "  %s/%s: P(high)=%.3f P(low)=%.3f",
            state["recent_steps_mean"],
            action,
            high_prob,
            low_prob,
        )


if __name__ == "__main__":
    main()
