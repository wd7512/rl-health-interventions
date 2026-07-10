"""Sprint 1 prompt rendering and generation.

Renders within-day and day-boundary prompts from the constant definitions in
prompts.py, then assembles them into the full bootstrap prompt list.

Algorithm 2 entry point: generate_prompts().
"""

from __future__ import annotations

import itertools

from rl_health_interventions.llm_bootstrapping.prompts.prompts import (
    ACTION_SENTENCES,
    ACTIONS,
    BIN_MIDPOINTS,
    BURDENS,
    DAY_TYPES,
    PERSONA_PROMPTS,
    SLEEP_TYPES,
    STEP_BIN_DISPLAY,
    STEP_BINS,
    TIMESTEP_NAMES,
)

_BURDEN_DESC = {
    "low": "0 interventions in last 3 timesteps",
    "medium": "1 intervention in last 3 timesteps",
    "high": "2-3 interventions in last 3 timesteps",
}


def _render_within_day(
    timestep: int,
    step_bin: str,
    burden: str,
    action: str,
    day_type: str,
    sleep: str,
) -> str:
    """Render a single within-day prompt for the given parameters."""
    time_name, prev_time_name = TIMESTEP_NAMES[timestep]
    act = ACTION_SENTENCES[action]
    if timestep == 0:
        prev_display = STEP_BIN_DISPLAY[step_bin]
        return (
            "# Current state\n"
            "You just woke up. It is the morning. "
            f"It is a {day_type}.\n"
            f"Last night you were {prev_display}.\n"
            f"Your sleep quality was {sleep}. "
            f"Your notification fatigue is {burden}.\n"
            f"{act}\n"
            "How many steps do you take this timestep?\n"
            'Output as: {"steps": N, "step_bin": "inactive"/"moderate"/"active"}'
        )
    prev_display = STEP_BIN_DISPLAY[step_bin]
    return (
        "# Current state\n"
        f"It is the {time_name}. Last timestep "
        f"({prev_time_name}) you were {prev_display}.\n"
        f"Your notification fatigue is {burden}. "
        f"It is a {day_type}.\n"
        f"Your sleep quality was {sleep}.\n"
        f"{act}\n"
        "How many steps do you take this timestep?\n"
        'Output as: {"steps": N, "step_bin": "inactive"/"moderate"/"active"}'
    )


def _render_day_boundary(
    step_bin_daily: str,
    burden: str,
    day_type: str,
    sleep: str,
) -> str:
    """Render a single day-boundary prompt."""
    midpoint = BIN_MIDPOINTS.get(step_bin_daily, 4000)
    steps_estimate = midpoint * 5
    return (
        "# Current state\n"
        f"You are at the end of the day. It was a {day_type}.\n"
        f"Your sleep quality last night was {sleep}. "
        f"Your notification fatigue is {burden} "
        f"({_BURDEN_DESC[burden]}).\n\n"
        f"Today you did {steps_estimate} total steps "
        f"({step_bin_daily}).\n\n"
        "It's bedtime. How was your sleep quality tonight?\n"
        '{"sleep_quality": "good"}\n'
        '{"sleep_quality": "poor"}'
    )


def _day_boundary_prompts() -> list[str]:
    """Generate all 36 unique day-boundary prompts."""
    combos = itertools.product(STEP_BINS, BURDENS, DAY_TYPES, SLEEP_TYPES)
    prompts = []
    for step_bin_daily, burden, day_type, sleep in combos:
        prompts.append(_render_day_boundary(step_bin_daily, burden, day_type, sleep))
    return prompts


def _within_day_prompts() -> list[str]:
    """Generate all 720 unique within-day prompts (144 x 5 timesteps)."""
    prompts = []
    for timestep in range(5):
        combos = itertools.product(
            STEP_BINS,
            BURDENS,
            ACTIONS,
            DAY_TYPES,
            SLEEP_TYPES,
        )
        for sb, b, a, d, s in combos:
            prompts.append(_render_within_day(timestep, sb, b, a, d, s))
    return prompts


def generate_prompts(
    persona: str = "base", samples_per_cell: int = 10
) -> tuple[str, list[str]]:
    """Return (system_prompt, list of prompt strings).

    Each prompt is repeated ``samples_per_cell`` times per output category
    (Algorithm 2, resolved-decisions-sprint-1.md).  Day-boundary prompts have
    2 output categories (good/poor) so they are repeated 2x; within-day prompts
    have 3 output categories so they are repeated 3x.

    Total: (36 x 20) + (720 x 30) = 22,320 prompts.
    """
    if persona not in PERSONA_PROMPTS:
        valid = list(PERSONA_PROMPTS.keys())
        msg = f"Unknown persona: {persona}. Must be one of {valid}"
        raise ValueError(msg)
    system_prompt = PERSONA_PROMPTS[persona]
    day_boundary = _day_boundary_prompts()
    within_day = _within_day_prompts()
    repeated = [p for p in day_boundary for _ in range(2 * samples_per_cell)] + [
        p for p in within_day for _ in range(3 * samples_per_cell)
    ]
    return system_prompt, repeated
