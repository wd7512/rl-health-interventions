"""PEARL prompt rendering and generation.

Separate from Sprint 1 prompts. PEARL uses 1 step/day, 13 actions
(idle + 12 COM-B x time), and burden tiers: none/minor/major.

The LLM simulates a 7-day walking history for a person in a given state
who receives a specific action. We then bin the raw step counts into
state factors deterministically.
"""

from __future__ import annotations

import itertools

# PEARL-specific burden tiers (matches YAML configs)
BURDENS = ["none", "minor", "major"]

_BURDEN_DESC = {
    "none": "0 interventions in the last 7 days",
    "minor": "1-3 interventions in the last 7 days",
    "major": "4+ interventions in the last 7 days",
}

# PEARL action space (12 COM-B x 2 time-of-day + idle)
ACTIONS = [
    "idle",
    "ability_morning",
    "ability_afternoon",
    "perceived_benefit_morning",
    "perceived_benefit_afternoon",
    "planning_morning",
    "planning_afternoon",
    "prioritization_morning",
    "prioritization_afternoon",
    "social_opportunity_morning",
    "social_opportunity_afternoon",
    "physical_opportunity_morning",
    "physical_opportunity_afternoon",
]

ACTION_DESCRIPTIONS = {
    "idle": "No intervention is delivered today.",
    "ability_morning": "Morning nudge to build walking ability.",
    "ability_afternoon": "Afternoon nudge to build walking ability.",
    "perceived_benefit_morning": "Morning nudge highlighting walking benefits.",
    "perceived_benefit_afternoon": "Afternoon nudge highlighting walking benefits.",
    "planning_morning": "Morning nudge about planning your walk.",
    "planning_afternoon": "Afternoon nudge about planning your walk.",
    "prioritization_morning": "Morning nudge about prioritizing walking.",
    "prioritization_afternoon": "Afternoon nudge about prioritizing walking.",
    "social_opportunity_morning": "Morning nudge about social walking.",
    "social_opportunity_afternoon": "Afternoon nudge about social walking.",
    "physical_opportunity_morning": "Morning nudge about weather/opportunity.",
    "physical_opportunity_afternoon": "Afternoon nudge about weather/opportunity.",
}

# PEARL state factors
RECENT_STEPS_MEAN = ["low", "moderate", "high"]
WALK_PATTERNS = ["low", "high"]
MORNING_RATIOS = ["morning", "balanced", "evening"]
DAY_TYPES = ["weekday", "weekend"]

# Binning thresholds for raw step counts (per PEARL Table 3)
STEPS_LOW_UPPER = 4000
STEPS_HIGH_LOWER = 7000
WALK_PATTERN_HIGH_THRESHOLD = 5000
MORNING_RATIO_LOW = 0.4
MORNING_RATIO_HIGH = 0.6

# Persona descriptions (from comb_scores.json)
PERSONA_DESCRIPTIONS = {
    "base": (
        "This person has moderate barriers across all COM-B dimensions. "
        "They are a morning person."
    ),
    "goal_driven": (
        "This person has high ability and motivation but needs planning support. "
        "They prefer afternoon activities."
    ),
    "social_responder": (
        "This person responds strongly to social opportunities but has low "
        "physical opportunity barriers. They prefer mornings."
    ),
    "stable_maintainer": (
        "This person has good ability and moderate barriers across the board. "
        "They have no strong time preference."
    ),
    "resistant": (
        "This person has high barriers across all COM-B dimensions. "
        "They are resistant to change and have no time preference."
    ),
}


def _render_system_prompt(persona: str) -> str:
    """Build the system prompt for a given persona."""
    persona_desc = PERSONA_DESCRIPTIONS.get(persona, PERSONA_DESCRIPTIONS["base"])
    return (
        "You are simulating a person's daily walking behavior in a health "
        "intervention study. The person receives daily notifications (nudges) "
        "designed to increase their physical activity.\n\n"
        f"Persona: {persona_desc}\n\n"
        "KEY FACTS:\n"
        "- Average person walks ~5,580 steps/day at baseline\n"
        "- Interventions (nudges) typically increase steps by 150-450 steps/day\n"
        "- Morning nudges are delivered at 6 AM, afternoon nudges at 3 PM\n"
        "- Walking patterns vary by time of day\n"
        "- Weekends typically have 5-20% fewer steps than weekdays\n\n"
        "You will simulate a 7-day walking history. For each day, provide:\n"
        "- morning_steps: steps taken before noon\n"
        "- afternoon_steps: steps taken after noon\n\n"
        "Be realistic: daily steps should vary (not be identical each day). "
        "Consider the person's state, the intervention received, and natural "
        "variation in walking behavior."
    )


def _render_user_prompt(
    recent_steps_mean: str,
    walk_pattern: str,
    morning_ratio: str,
    day_type: str,
    burden: str,
    action: str,
) -> str:
    """Build a user prompt for one (state, action) combination."""
    burden_desc = _BURDEN_DESC.get(burden, f"Burden: {burden}")
    action_desc = ACTION_DESCRIPTIONS.get(action, f"Action: {action}")

    # Describe current state in natural language
    steps_desc = {
        "low": "averaging under 4,000 steps/day recently",
        "moderate": "averaging 4,000-7,000 steps/day recently",
        "high": "averaging over 7,000 steps/day recently",
    }
    walk_desc = {
        "low": "tending to walk less overall",
        "high": "tending to walk more overall",
    }
    ratio_desc = {
        "morning": "doing most of their walking in the morning",
        "balanced": "distributing walking evenly throughout the day",
        "evening": "doing most of their walking in the evening/afternoon",
    }

    return (
        f"# Scenario\n"
        f"A person is in the following state:\n"
        f"- Recent activity: {steps_desc[recent_steps_mean]}\n"
        f"- Walk pattern: {walk_desc[walk_pattern]}\n"
        f"- Time of day preference: {ratio_desc[morning_ratio]}\n"
        f"- Day type: {day_type}\n"
        f"- Notification fatigue: {burden_desc}\n\n"
        f"# Intervention\n"
        f"{action_desc}\n\n"
        f"# Task\n"
        f"Simulate this person's walking for the next 7 days. "
        f"For each day, provide morning_steps and afternoon_steps.\n\n"
        f"Output exactly 7 JSON objects, one per day:\n"
        f'{{"day": 1, "morning_steps": N, "afternoon_steps": N}}\n'
        f'{{"day": 2, "morning_steps": N, "afternoon_steps": N}}\n'
        f"...\n"
        f'{{"day": 7, "morning_steps": N, "afternoon_steps": N}}'
    )


PromptEntry = tuple[str, dict, str]  # (prompt_text, state, action)


def generate_prompts(
    persona: str = "base",
    samples_per_cell: int = 10,
    state_subset: list[dict] | None = None,
) -> tuple[str, list[PromptEntry]]:
    """Return (system_prompt, list of PromptEntry tuples).

    Each entry pairs the prompt string with its originating state and action
    metadata, so downstream aggregation never depends on loop ordering.

    Parameters
    ----------
    persona : str
        Persona name for the system prompt.
    samples_per_cell : int
        Number of LLM samples per (state, action) cell.
    state_subset : list[dict] | None
        Optional list of state dicts to generate prompts for. If None,
        generates for all 108 states x 13 actions = 1,404 combinations.
    """
    system_prompt = _render_system_prompt(persona)

    if state_subset is not None:
        state_action_combos = [
            (state, action) for state in state_subset for action in ACTIONS
        ]
    else:
        all_states = [
            {
                "recent_steps_mean": rsm,
                "recent_walk_pattern": wp,
                "morning_steps_ratio": mr,
                "day_of_week": dt,
                "burden": b,
            }
            for rsm, wp, mr, dt, b in itertools.product(
                RECENT_STEPS_MEAN, WALK_PATTERNS, MORNING_RATIOS, DAY_TYPES, BURDENS
            )
        ]
        state_action_combos = [
            (state, action) for state in all_states for action in ACTIONS
        ]

    prompts: list[PromptEntry] = []
    for state, action in state_action_combos:
        prompt = _render_user_prompt(
            recent_steps_mean=state["recent_steps_mean"],
            walk_pattern=state["recent_walk_pattern"],
            morning_ratio=state["morning_steps_ratio"],
            day_type=state["day_of_week"],
            burden=state["burden"],
            action=action,
        )
        for _ in range(samples_per_cell):
            prompts.append((prompt, state, action))

    return system_prompt, prompts
