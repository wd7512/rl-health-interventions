"""PEARL prompt rendering and generation (stub).

Separate from Sprint 1 prompts. PEARL uses 1 step/day, 13 actions
(idle + 12 COM-B x time), and burden tiers: none/minor/major.

TODO(#282): Implement full PEARL prompt rendering.
"""

from __future__ import annotations

# PEARL-specific burden tiers (matches YAML configs)
BURDENS = ["none", "minor", "major"]

_BURDEN_DESC = {
    "none": "0 interventions in last 7 days",
    "minor": "1-3 interventions in last 7 days",
    "major": "4+ interventions in last 7 days",
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

# PEARL state factors
STEP_BINS = ["inactive", "moderate", "active"]
WALK_PATTERNS = ["low", "high"]
MORNING_RATIOS = ["morning", "balanced", "evening"]
DAY_TYPES = ["weekday", "weekend"]
SLEEP_TYPES = ["good", "poor"]


def generate_prompts(
    persona: str = "base",
    samples_per_cell: int = 10,  # noqa: ARG001
) -> tuple[str, list[str]]:
    """Return (system_prompt, list of prompt strings).

    TODO(#282): Implement full PEARL prompt generation.
    For now, return empty list as a stub.
    """
    # TODO: Build system prompt for persona
    # TODO: Generate all state x action combinations
    # TODO: Render within-day prompts (1 step/day)
    system_prompt = f"TODO: PEARL system prompt for {persona}"
    return system_prompt, []
