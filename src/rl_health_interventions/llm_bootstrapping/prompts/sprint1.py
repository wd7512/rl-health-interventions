"""Sprint 1 prompt definitions for bootstrap transition tables.

Matches resolved-decisions-sprint-1.md §LLM bootstrapping prompt design.
"""

from __future__ import annotations

import itertools

SYSTEM_PROMPT = """\
# Reference

You are a generally healthy adult looking to improve your exercise and sleep habits.

5 timesteps per day: morning, mid-morning, lunch, afternoon, evening

Per-timestep step ranges (daily threshold / 5):
  <800 steps     = inactive
  800-1600 steps = moderate
  >1600 steps    = active

Sleep quality: good / poor (based on how well you slept)

Daily step total ranges (5 timesteps x per-timestep ranges):
  <4000 steps total     = inactive
  4000-8000 steps total = moderate
  >8000 steps total     = active

Burden (notification fatigue):
  low     = 0 of last 3 timesteps had an intervention
  medium  = 1 of last 3
  high    = 2 or 3 of last 3"""

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

# Decisions doc uses "moderately active" for previous step bin display
STEP_BIN_DISPLAY = {
    "inactive": "inactive",
    "moderate": "moderately active",
    "active": "active",
}

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


def _render_within_day(timestep, step_bin, burden, action, day_type, sleep):
    time_name, prev_time_name = TIMESTEP_NAMES[timestep]
    act = ACTION_SENTENCES[action]
    if timestep == 0:
        return (
            "# Current state\n"
            "You just woke up. It is the morning. "
            f"It is a {day_type}.\n"
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


def _render_day_boundary(step_bin_daily, burden, day_type, sleep, notifications):
    midpoint = BIN_MIDPOINTS.get(step_bin_daily, 4000)
    steps_estimate = midpoint * 5
    return (
        "# Current state\n"
        f"You are at the end of the day. It was a {day_type}.\n"
        f"Your sleep quality last night was {sleep}. "
        f"Your notification fatigue is {burden}.\n\n"
        f"Today you did {steps_estimate} total steps "
        f"({step_bin_daily}).\n"
        f"Your notifications today: {notifications}\n\n"
        "It's bedtime. How was your sleep quality tonight?\n"
        '{"sleep_quality": "good"}\n'
        '{"sleep_quality": "poor"}'
    )


def _day_boundary_prompts():
    combos = itertools.product(STEP_BINS, BURDENS, DAY_TYPES, SLEEP_TYPES)
    prompts = []
    for step_bin_daily, burden, day_type, sleep in combos:
        # Notification sequence depends on burden level
        notifications = NOTIFICATION_SEQUENCES[burden]
        prompts.append(
            _render_day_boundary(
                step_bin_daily,
                burden,
                day_type,
                sleep,
                notifications,
            )
        )
    return prompts


def _within_day_prompts():
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


def generate_prompts():
    """Return (system_prompt, list of prompt strings)."""
    return SYSTEM_PROMPT, _day_boundary_prompts() + _within_day_prompts()
