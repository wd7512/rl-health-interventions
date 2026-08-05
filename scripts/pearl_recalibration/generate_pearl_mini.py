"""Generate a mini PEARL transition table using LLM bootstrapping.

Pilot script: generates a small table (~4 states x 13 actions = 52 cells)
to validate the pipeline end-to-end before scaling to full 108-state table.

States include all 5 MDP factors (recent_steps_mean, recent_walk_pattern,
morning_steps_ratio, day_of_week, burden).

Usage: uv run python scripts/pearl_recalibration/generate_pearl_mini.py
    [samples_per_cell] [prompt_variant]

prompt_variant selects the prompt style (see prompts.pearl.PROMPT_VARIANTS;
default "baseline"). Output table: tables/pearl_12action_pilot/
pearl_pilot{_<variant>}.json, raw results in tables/pearl_12action_pilot/raw/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
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
    parse_day_history,
)
from rl_health_interventions.llm_bootstrapping.prompts.pearl import (  # noqa: E402
    ACTIONS,
    PROMPT_VARIANTS,
    generate_prompts,
)
from rl_health_interventions.llm_bootstrapping.request import (  # noqa: E402
    DEFAULT_TEMPERATURE,
    batch_complete,
)
from rl_health_interventions.llm_bootstrapping.table_aggregate import (  # noqa: E402
    aggregate_to_table,
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


def _resolve_temperature(temperature: float | None) -> float:
    """Resolve the effective sampling temperature.

    None means "use request.batch_complete's default"
    (request.DEFAULT_TEMPERATURE); an explicit value passes through unchanged.
    """
    return temperature if temperature is not None else DEFAULT_TEMPERATURE


def _request(
    prompts: list[str],
    system_prompt: str,
    temperature: float | None,
    timeout: float | None,
) -> list[dict]:
    """Run batch_complete forwarding the resolved temperature and timeout.

    Extracted from main so the initial and retry calls share one path (and
    one resolved temperature) and are directly testable.
    """
    return batch_complete(
        prompts,
        system_prompt=system_prompt,
        temperature=_resolve_temperature(temperature),
        max_workers=50,
        provider="openrouter",
        timeout=timeout,
    )


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Generate the pilot PEARL transition table via LLM bootstrapping."""
    setup_logging()
    load_env()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", nargs="?", type=int, default=SAMPLES_PER_CELL)
    parser.add_argument(
        "variant", nargs="?", default="baseline", choices=PROMPT_VARIANTS
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: request.batch_complete default)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-request timeout in seconds (default: litellm's default); "
        "a short value makes hung requests fail fast",
    )
    args = parser.parse_args()
    variant, samples = args.variant, args.samples
    temperature = args.temperature
    timeout = args.timeout

    out_dir = Path(_REPO_ROOT / "tables" / "pearl_12action_pilot")
    out_dir.mkdir(parents=True, exist_ok=True)
    table_name = (
        "pearl_pilot.json" if variant == "baseline" else f"pearl_pilot_{variant}.json"
    )
    out_path = out_dir / table_name

    logger.info(
        "Generating mini-table (variant=%s): %d states x %d actions x %d samples",
        variant,
        len(MINI_STATES),
        len(ACTIONS),
        samples,
    )

    system_prompt, prompt_entries = generate_prompts(
        persona="base",
        samples_per_cell=samples,
        state_subset=MINI_STATES,
        prompt_variant=variant,
    )
    logger.info("Generated %d prompts", len(prompt_entries))

    # Call LLM
    effective_temperature = _resolve_temperature(temperature)
    logger.info("Calling LLM (temperature=%s)...", effective_temperature)
    results = _request(
        [p for p, _s, _a in prompt_entries],
        system_prompt,
        temperature,
        timeout,
    )

    # Count successes
    ok = sum(1 for r in results if "content" in r)
    logger.info("LLM results: %d/%d succeeded", ok, len(results))

    # Bounded retry for unparseable responses (Option B: retry-on-parse-None).
    # One retry per record; the original (unparseable) attempt is preserved in
    # the raw file alongside the retry so the parse-failure rate stays diagnosable.
    state_action_pairs = [(s, a) for _p, s, a in prompt_entries]
    retry_indices = [
        i
        for i, (result, (state, action)) in enumerate(
            zip(results, state_action_pairs, strict=True)
        )
        if "error" not in result
        and parse_day_history(result.get("content", "")) is None
    ]
    retried_originals: dict[int, dict] = {}
    if retry_indices:
        logger.info("Retrying %d unparseable response(s)", len(retry_indices))
        retry_results = _request(
            [prompt_entries[i][0] for i in retry_indices],
            system_prompt,
            temperature,
            timeout,
        )
        for idx, retry_result in zip(retry_indices, retry_results, strict=True):
            retried_originals[idx] = results[idx]
            results[idx] = retry_result
        n_ok_after = sum(
            1
            for i in retry_indices
            if "content" in results[i]
            and parse_day_history(results[i]["content"]) is not None
        )
        logger.info("Retry recovered %d/%d records", n_ok_after, len(retry_indices))

    # Save raw results for diagnosis (parse failures, bad output inspection).
    # Retried records keep their original attempt under "original_content" (or
    # "original_error") alongside the retry, so both attempts are preserved.
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"results_{variant}_{datetime.now(UTC):%Y%m%d_%H%M%S}.jsonl"
    with raw_path.open("w") as f:
        for idx, (result, (state, action)) in enumerate(
            zip(results, state_action_pairs, strict=True)
        ):
            record = {"state": state, "action": action}
            if "error" in result:
                record["error"] = result["error"]
            else:
                record["content"] = result["content"]
            if idx in retried_originals:
                original = retried_originals[idx]
                if "error" in original:
                    record["original_error"] = original["error"]
                else:
                    record["original_content"] = original["content"]
            f.write(json.dumps(record) + "\n")
    logger.info("Saved %d raw results to %s", len(results), raw_path)

    # Aggregate — use metadata embedded in prompt_entries, not reconstructed
    table = aggregate_to_table(results, state_action_pairs)
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
