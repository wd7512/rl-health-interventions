"""Sprint 1 prompt definitions for bootstrap transition tables."""

from __future__ import annotations

import itertools

SYSTEM_PROMPT = (
    "Rules: steps <800=inactive, 800-1600=moderate, >1600=active "
    "per timestep. Daily <4k=inactive, 4-8k=moderate, >8k=active. "
    "Sleep=good|poor. Burden=low|medium|high."
)

STEP_BINS = ["inactive", "moderate", "active"]
BURDENS = ["low", "medium", "high"]
ACTIONS = ["idle", "movement_suggestion", "goal_reminder", "journal"]
DAY_TYPES = ["weekday", "weekend"]
SLEEP_TYPES = ["good", "poor"]

TIMESTEP_NAMES = [
    ("morning", None),
    ("mid-morning", "morning"),
    ("lunch", "mid-morning"),
    ("afternoon", "lunch"),
    ("evening", "afternoon"),
]

ACTION_SENTENCES = {
    "idle": "",
    "movement_suggestion": "Your phone buzzes with a movement suggestion.",
    "goal_reminder": "Your phone reminds you of your step goal.",
    "journal": "Your phone prompts you to write in your journal.",
}

NOTIFICATION_SEQUENCES = {
    "low": "idle, idle, idle, idle, idle",
    "medium": "movement_suggestion, idle, idle, idle, idle",
    "high": "movement_suggestion, goal_reminder, idle, idle, idle",
}

BIN_MIDPOINTS = {"inactive": 400, "moderate": 1200, "active": 2000}


def _render_within_day(
    timestep, step_bin, burden, action, day_type, sleep
):
    time_name, prev_time_name = TIMESTEP_NAMES[timestep]
    act = ACTION_SENTENCES[action]
    if timestep == 0:
        return (
            f"# Current state\n"
            f"You just woke up. It is the morning. "
            f"It is a {day_type}.\n"
            f"Your sleep quality was {sleep}. "
            f"Your notification fatigue is {burden}.\n"
            f"{act}\n"
            f"How many steps do you take this timestep?"
        )
    return (
        f"# Current state\n"
        f"It is the {time_name}. Last timestep "
        f"({prev_time_name}) you were {step_bin}.\n"
        f"Your notification fatigue is {burden}. "
        f"It is a {day_type}.\n"
        f"Your sleep quality was {sleep}.\n"
        f"{act}\n"
        f"How many steps do you take this timestep?"
    )


def _render_day_boundary(step_bin_daily, burden, day_type, sleep):
    midpoint = BIN_MIDPOINTS.get(step_bin_daily, 4000)
    steps_estimate = midpoint * 5
    seq = NOTIFICATION_SEQUENCES.get(
        burden, "idle, idle, idle, idle, idle"
    )
    return (
        f"# Current state\n"
        f"You are at the end of the day. It was a {day_type}.\n"
        f"Your sleep quality last night was {sleep}. "
        f"Your notification fatigue is {burden}.\n"
        f"Today you did ~{steps_estimate} total steps "
        f"({step_bin_daily}).\n"
        f"Your notifications today: {seq}\n\n"
        f"It's bedtime. How was your sleep quality tonight?"
    )


def _day_boundary_prompts():
    combos = itertools.product(
        STEP_BINS, BURDENS, DAY_TYPES, SLEEP_TYPES
    )
    return [
        _render_day_boundary(sbd, b, d, s)
        for sbd, b, d, s in combos
    ]


def _within_day_prompts():
    prompts = []
    for timestep in range(5):
        combos = itertools.product(
            STEP_BINS, BURDENS, ACTIONS, DAY_TYPES, SLEEP_TYPES
        )
        for sb, b, a, d, s in combos:
            prompts.append(
                _render_within_day(timestep, sb, b, a, d, s)
            )
    return prompts


def generate_prompts():
    """Return (system_prompt, list of prompt strings)."""
    return SYSTEM_PROMPT, _day_boundary_prompts() + _within_day_prompts()
