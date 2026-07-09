"""Prompt string constants for Sprint 1 bootstrap.

These define the constant values used in prompt rendering: state dimensions,
action descriptions, bin thresholds, and the system prompt.

Should not contain rendering or generation logic — that lives in sprint1.py.
"""

from __future__ import annotations

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

BIN_MIDPOINTS = {"inactive": 400, "moderate": 1200, "active": 2000}
