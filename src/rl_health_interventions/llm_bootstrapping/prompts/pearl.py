"""PEARL prompt rendering and generation.

Separate from Sprint 1 prompts. PEARL uses 1 step/day, 13 actions
(idle + 12 COM-B x time), and burden tiers: none/minor/major.

The LLM simulates a 7-day walking history for a person in a given state
who receives a specific action. We then bin the raw step counts into
state factors deterministically.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

# Prompt variants: ladder from implicit (baseline) to fully explicit
# alignment with the PEARL RCT paper. Each round of the refinement log
# (docs/research/prompt-refinement-log.md) uses one variant.
# Empirical facts source:
# docs/research/recreations/pearl-rct-2025/pearl-deep-analysis.md
PROMPT_VARIANTS = (
    "baseline",
    "state_self_model",
    "com_b_mechanisms",
    "empirical_anchors",
    "protocol",
    "protocol_fewshot",
)


class PromptVariant:
    """Prompt text blocks for one variant of the PEARL bootstrapping prompt.

    Parameters
    ----------
    system_extra : str
        Text appended to the base system prompt.
    action_overrides : dict[str, str] | None
        Replacement per-action sentences (keyed by action name).
    user_extra : str | Callable[[dict, str], str] | None
        Text appended to every user prompt, or a callable returning it
        given the (state, action) pair.
    """

    def __init__(
        self,
        *,
        system_extra: str = "",
        action_overrides: dict[str, str] | None = None,
        user_extra: str | Callable[[dict, str], str] | None = None,
    ) -> None:
        self.system_extra = system_extra
        self.action_overrides = action_overrides or {}
        self.user_extra = user_extra

    def render_user_extra(self, state: dict, action: str) -> str:
        """Return the variant's user-prompt extra text for this cell."""
        user_extra = self.user_extra
        if isinstance(user_extra, str):
            return user_extra
        if user_extra is not None:
            return user_extra(state, action)
        return ""


# State-conditional self-model anchors for the "state_self_model" variant.
# The recent_steps_mean band stated in each scenario is the person's true
# recent daily average; the population baseline (~5,580) is only an average
# and must not override it.
_STATE_STEP_ANCHORS = {
    "low": 3000,
    "moderate": 5500,
    "high": 8000,
}

_STATE_RATIO_SHARE_DESC = {
    "morning": "about 60-75% of their steps before noon",
    "balanced": "roughly 40-60% of their steps before noon",
    "evening": (
        "only about 25-40% of their steps before noon (most walking happens "
        "in the afternoon/evening)"
    ),
}

_STATE_BURDEN_EFFECT_DESC = {
    "none": (
        "the person is fully receptive, so the day's boost tends toward the "
        "top of that range (+300-450 steps)"
    ),
    "minor": (
        "the person is somewhat receptive, so the day's boost tends toward "
        "the middle of that range (+250-400 steps)"
    ),
    "major": (
        "the person is fatigued by notifications, so the day's boost tends "
        "toward the bottom of that range (+150-250 steps)"
    ),
}

_SELF_MODEL_SYSTEM_EXTRA = (
    "SELF-MODEL NOTE: the ~5,580 steps/day figure above is only the "
    "population average at baseline - it is NOT this person's level. Every "
    "scenario states the person's CURRENT recent activity level explicitly "
    "(recent activity: low = under 4,000 steps/day, moderate = 4,000-7,000, "
    "high = over 7,000 steps/day). Treat the stated band as this person's "
    "true recent average and keep all 7 simulated days consistent with it: a "
    "'high' person's days stay well above 7,000, a 'low' person's days stay "
    "well below 4,000. On days with no intervention (idle), the person simply "
    "continues at their own established level; never regress a stated 'high' "
    "or 'low' person toward the population average."
)


def _state_self_model_user_extra(state: dict, action: str) -> str:
    """State-conditional step anchors for the state_self_model variant."""
    rsm = state["recent_steps_mean"]
    anchor = _STATE_STEP_ANCHORS.get(rsm, 5500)
    ratio_desc = _STATE_RATIO_SHARE_DESC.get(
        state["morning_steps_ratio"], _STATE_RATIO_SHARE_DESC["balanced"]
    )
    burden_desc = _STATE_BURDEN_EFFECT_DESC.get(
        state["burden"], _STATE_BURDEN_EFFECT_DESC["none"]
    )
    if action == "idle":
        day_line = (
            "Today is an idle day (no intervention): the person walks at "
            "their own established level with no step boost."
        )
    else:
        day_line = (
            "Today's intervention is delivered: add roughly 150-450 steps on "
            f"top of the person's own level ({burden_desc})."
        )
    return (
        "SELF-MODEL ANCHORS: this person's recent daily average is around "
        f"{anchor:,} steps/day (recent activity: {rsm}). Keep each day's "
        f"total within roughly +/-500 steps of {anchor:,} - daily totals "
        "should hover around that level and never drift toward the "
        f"population average. {day_line} The person's time-of-day preference "
        f"is {state['morning_steps_ratio']}: {ratio_desc}; a morning "
        "intervention leans the day toward the upper end of that share and "
        "an afternoon intervention toward the lower end."
    )


PROMPT_VARIANT_CONFIGS: dict[str, PromptVariant] = {
    "baseline": PromptVariant(),
    "state_self_model": PromptVariant(
        system_extra=_SELF_MODEL_SYSTEM_EXTRA,
        user_extra=_state_self_model_user_extra,
    ),
    "com_b_mechanisms": PromptVariant(),
    "empirical_anchors": PromptVariant(),
    "protocol": PromptVariant(),
    "protocol_fewshot": PromptVariant(),
}

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


def _render_system_prompt(persona: str, prompt_variant: str = "baseline") -> str:
    """Build the system prompt for a given persona (and optional variant)."""
    persona_desc = PERSONA_DESCRIPTIONS.get(persona, PERSONA_DESCRIPTIONS["base"])
    base = (
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
    extra = PROMPT_VARIANT_CONFIGS[prompt_variant].system_extra
    if extra:
        base += f"\n\n{extra}"
    return base


def _render_user_prompt(
    recent_steps_mean: str,
    walk_pattern: str,
    morning_ratio: str,
    day_type: str,
    burden: str,
    action: str,
    prompt_variant: str = "baseline",
) -> str:
    """Build a user prompt for one (state, action) combination."""
    variant = PROMPT_VARIANT_CONFIGS[prompt_variant]
    burden_desc = _BURDEN_DESC.get(burden, f"Burden: {burden}")
    action_desc = variant.action_overrides.get(
        action, ACTION_DESCRIPTIONS.get(action, f"Action: {action}")
    )

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

    state = {
        "recent_steps_mean": recent_steps_mean,
        "recent_walk_pattern": walk_pattern,
        "morning_steps_ratio": morning_ratio,
        "day_of_week": day_type,
        "burden": burden,
    }
    base = (
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
    )
    extra = variant.render_user_extra(state, action)
    if extra:
        base += f"{extra}\n\n"
    base += (
        "Output exactly 7 JSON objects, one per day:\n"
        '{"day": 1, "morning_steps": N, "afternoon_steps": N}\n'
        '{"day": 2, "morning_steps": N, "afternoon_steps": N}\n'
        "...\n"
        '{"day": 7, "morning_steps": N, "afternoon_steps": N}'
    )
    return base


PromptEntry = tuple[str, dict, str]  # (prompt_text, state, action)


def generate_prompts(
    persona: str = "base",
    samples_per_cell: int = 10,
    state_subset: list[dict] | None = None,
    prompt_variant: str = "baseline",
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
    prompt_variant : str
        Prompt variant name (see PROMPT_VARIANTS). Defaults to "baseline".
    """
    if prompt_variant not in PROMPT_VARIANT_CONFIGS:
        msg = (
            f"Unknown prompt_variant {prompt_variant!r}; choose from {PROMPT_VARIANTS}"
        )
        raise ValueError(msg)

    system_prompt = _render_system_prompt(persona, prompt_variant)

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
            prompt_variant=prompt_variant,
        )
        for _ in range(samples_per_cell):
            prompts.append((prompt, state, action))

    return system_prompt, prompts
