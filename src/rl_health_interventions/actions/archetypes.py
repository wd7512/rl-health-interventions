from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Intervention:
    name: str
    archetype: str
    description: str
    base_cost: float


MOVEMENT_SUGGESTION = Intervention(
    "movement_suggestion", "movement", "Suggest going for a walk", 0.03
)
STEP_GOAL = Intervention("step_goal", "movement", "Set a daily step goal", 0.04)
WALK_REMINDER = Intervention("walk_reminder", "movement", "Remind to take a walk", 0.02)
EXERCISE_NUDGE = Intervention("exercise_nudge", "movement", "Nudge to exercise", 0.05)
ACTIVE_BREAK = Intervention("active_break", "movement", "Take an active break", 0.02)

JOURNAL = Intervention("journal", "cognitive", "Write in journal", 0.04)
REFLECTION_PROMPT = Intervention(
    "reflection_prompt", "cognitive", "Reflect on the day", 0.03
)
GRATITUDE_NUDGE = Intervention(
    "gratitude_nudge", "cognitive", "Practice gratitude", 0.02
)
MOOD_CHECK = Intervention("mood_check", "cognitive", "Check your mood", 0.02)
GOAL_REMINDER = Intervention("goal_reminder", "cognitive", "Review your goals", 0.03)

SLEEP_HYGIENE = Intervention("sleep_hygiene", "wellness", "Sleep hygiene tips", 0.03)
HYDRATION_REMINDER = Intervention("hydration_reminder", "wellness", "Drink water", 0.02)
BREATHING_EXERCISE = Intervention(
    "breathing_exercise", "wellness", "Breathing exercise", 0.03
)
STRESS_CHECK = Intervention("stress_check", "wellness", "Check stress level", 0.02)
WIND_DOWN = Intervention("wind_down", "wellness", "Wind down routine", 0.03)

ALL_INTERVENTIONS: list[Intervention] = [
    MOVEMENT_SUGGESTION,
    STEP_GOAL,
    WALK_REMINDER,
    EXERCISE_NUDGE,
    ACTIVE_BREAK,
    JOURNAL,
    REFLECTION_PROMPT,
    GRATITUDE_NUDGE,
    MOOD_CHECK,
    GOAL_REMINDER,
    SLEEP_HYGIENE,
    HYDRATION_REMINDER,
    BREATHING_EXERCISE,
    STRESS_CHECK,
    WIND_DOWN,
]

ARCHETYPES: dict[str, list[Intervention]] = {
    "movement": [
        MOVEMENT_SUGGESTION,
        STEP_GOAL,
        WALK_REMINDER,
        EXERCISE_NUDGE,
        ACTIVE_BREAK,
    ],
    "cognitive": [
        JOURNAL,
        REFLECTION_PROMPT,
        GRATITUDE_NUDGE,
        MOOD_CHECK,
        GOAL_REMINDER,
    ],
    "wellness": [
        SLEEP_HYGIENE,
        HYDRATION_REMINDER,
        BREATHING_EXERCISE,
        STRESS_CHECK,
        WIND_DOWN,
    ],
}

ARCHETYPE_NAMES = ["movement", "cognitive", "wellness"]
ARCHETYPE_TO_INDEX = {name: i for i, name in enumerate(ARCHETYPE_NAMES)}

ACTION_TO_ARCHETYPE: dict[str, str] = {i.name: i.archetype for i in ALL_INTERVENTIONS}


def get_intervention_names() -> list[str]:
    return [i.name for i in ALL_INTERVENTIONS]


def get_intervention_by_name(name: str) -> Intervention | None:
    for i in ALL_INTERVENTIONS:
        if i.name == name:
            return i
    return None
