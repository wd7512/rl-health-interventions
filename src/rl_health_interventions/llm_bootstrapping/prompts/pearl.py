"""PEARL prompt rendering and generation.

Separate from Sprint 1 prompts. PEARL uses 1 step/day, 13 actions
(idle + 12 COM-B x time), and burden tiers: none/minor/major.

The LLM simulates a 7-day walking history for a person in a given state
who receives a specific action. We then bin the raw step counts into
state factors deterministically.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Prompt variants: ladder from implicit (baseline) to fully explicit
# alignment with the PEARL RCT paper. Each round of the refinement log
# (docs/research/prompt-refinement-log.md) uses one variant.
# Empirical facts source:
# docs/research/recreations/pearl-rct-2025/pearl-deep-analysis.md


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


# COM-B mechanism framing for the "com_b_mechanisms" variant. Each nudge is
# given a causal mechanism: which behavioural lever it targets and why that
# lever matters for this person. Matched nudges (target a lever the person
# has a barrier on) produce the +150-450 step boost; unmatched nudges are
# near no-ops. The persistence rule from round 2 is kept so idle baselines
# stay at the person's stated level (the step-lift metric needs it).
_COMB_MECHANISMS_SYSTEM_EXTRA = (
    "COM-B MECHANISMS: behaviour change works through levers. Capability "
    "is knowing how to walk and finding it easy; Opportunity is whether the "
    "environment, schedule, and other people make walking possible; "
    "Motivation is wanting to walk and believing it matters; self-regulation "
    "(planning and prioritization) turns an intention into a concrete, "
    "scheduled act.\n\n"
    "CAUSAL RULE: every nudge targets one lever. A nudge that matches a "
    "lever the person has a barrier on produces a REAL increase in walking "
    "on the day it is delivered (+150-450 steps). A nudge that does not "
    "match the person's barriers produces only a small or negligible "
    "increase: someone who already knows how to walk is not moved by another "
    "ability tip, and someone with no practical obstacle is not moved by an "
    "opportunity reminder.\n\n"
    "THEME MAPPING: ability = capability; physical_opportunity = "
    "opportunity (the environment); social_opportunity = opportunity via "
    "other people; perceived_benefit = reflective motivation; planning and "
    "prioritization = self-regulation.\n\n"
    "PERSISTENCE: each scenario states the person's current recent activity "
    "level (low = under 4,000, moderate = 4,000-7,000, high = over 7,000 "
    "steps/day). Days without an effective nudge (idle, or an unmatched "
    "nudge) keep the person at their own established level - never regress "
    "a stated 'high' or 'low' person toward the ~5,580 population average."
)

_COMB_MECHANISMS_ACTION_OVERRIDES = {
    "idle": "No intervention is delivered today.",
    "ability_morning": (
        "A morning message offers a technique that makes walking easier "
        "(e.g. a shorter route, or a walk paired with your morning coffee); "
        "receiving it, you find walking less effortful and take a noticeably "
        "longer walk this morning."
    ),
    "ability_afternoon": (
        "An afternoon message offers a technique that makes walking easier "
        "(e.g. splitting one long walk into two short ones); receiving it, "
        "you find walking less effortful and take a noticeably longer walk "
        "this afternoon."
    ),
    "perceived_benefit_morning": (
        "A morning message reminds you what walking does for you (e.g. more "
        "energy and a better mood for the day); it sharpens your motivation, "
        "so you set out more readily and take a noticeably longer walk this "
        "morning."
    ),
    "perceived_benefit_afternoon": (
        "An afternoon message reminds you what walking does for you (e.g. a "
        "head-clearing break that helps you sleep); it sharpens your "
        "motivation, so you set out more readily and take a noticeably "
        "longer walk this afternoon."
    ),
    "planning_morning": (
        "A morning message helps you schedule today's walk (e.g. a specific "
        "time and route written down before the day fills up); the concrete "
        "plan turns intention into action, so you take a noticeably longer "
        "walk this morning."
    ),
    "planning_afternoon": (
        "An afternoon message helps you schedule the rest of the day (e.g. "
        "blocking out a specific time and route); the concrete plan turns "
        "intention into action, so you take a noticeably longer walk this "
        "afternoon."
    ),
    "prioritization_morning": (
        "A morning message helps you protect time for walking (e.g. moving "
        "a low-priority task off the morning); walking becomes a deliberate "
        "priority, so you take a noticeably longer walk this morning."
    ),
    "prioritization_afternoon": (
        "An afternoon message helps you protect time for walking (e.g. "
        "delaying a low-priority task to make room); walking becomes a "
        "deliberate priority, so you take a noticeably longer walk this "
        "afternoon."
    ),
    "social_opportunity_morning": (
        "A morning message connects walking to other people (e.g. a friend "
        "who will join you, or a walk-and-talk call); the social opening "
        "gets you out the door, so you take a noticeably longer walk this "
        "morning."
    ),
    "social_opportunity_afternoon": (
        "An afternoon message connects walking to other people (e.g. a group "
        "walk or a colleague to join); the social opening gets you out the "
        "door, so you take a noticeably longer walk this afternoon."
    ),
    "physical_opportunity_morning": (
        "A morning message points to a physical opening (e.g. favourable "
        "weather or a pleasant nearby route); the practical opportunity "
        "makes the walk easier to take, so you take a noticeably longer walk "
        "this morning."
    ),
    "physical_opportunity_afternoon": (
        "An afternoon message points to a physical opening (e.g. a quieter "
        "path or a stop on the way home); the practical opportunity makes "
        "the walk easier to take, so you take a noticeably longer walk this "
        "afternoon."
    ),
}

_COMB_MECHANISMS_USER_EXTRA = (
    "Respond to the intervention if it fits your situation: a matched nudge "
    "produces a clear increase in steps that day (~150-450 steps); an "
    "unmatched or no nudge leaves your steps near your usual level."
)


# Round 4: "empirical_anchors" — the synthesis. Restores the round-2 style
# explicit numeric self-model anchors (persistence), adds a per-state
# barrier profile so every persona has at least one real barrier that a
# matched nudge relieves (fixes round 3's near-zero high/none lift), keeps
# the round-3 COM-B mechanism framing, and adds the empirical anchor from
# the PEARL RCT paper (ability messages best received at 90% thumbs-up).
_EMPIRICAL_BARRIER_PROFILES = {
    ("low", "none"): (
        "a struggling walker with no extra load - the barrier is effort and "
        "self-efficacy: walking feels hard and they doubt they can do it"
    ),
    ("low", "major"): (
        "a struggling walker under heavy strain - the barriers are fatigue "
        "AND opportunity: the strain drains their energy and fills their time"
    ),
    ("high", "none"): (
        "an active person with no extra load - the barriers are complacency "
        "and reinforcement: the walking habit is there but needs keeping"
    ),
    ("high", "major"): (
        "an active person under heavy strain - the barrier is scheduling and "
        "fatigue, NOT capability: they know how to walk, but the strain "
        "disrupts their routine"
    ),
}

_EMPIRICAL_ANCHORS_SYSTEM_EXTRA = (
    "SELF-MODEL NOTE: the ~5,580 steps/day figure above is only the "
    "population average at baseline - it is NOT this person's level. Every "
    "scenario states the person's CURRENT recent activity level explicitly "
    "(recent activity: low = under 4,000 steps/day, moderate = 4,000-7,000, "
    "high = over 7,000 steps/day). Without intervention the person stays at "
    "their current level: a 'high' person keeps walking around 8,000 steps "
    "per day, a 'low' person around 3,000. On days with no intervention "
    "(idle), the person simply continues at their own established level; "
    "never regress a stated 'high' or 'low' person toward the population "
    "average.\n\n"
    "BARRIER PROFILE: each day the person falls into one of four profiles. "
    "A struggling walker with no extra load (low activity, no burden) faces "
    "an effort/self-efficacy barrier. A struggling walker under heavy "
    "strain (low activity, major burden) faces fatigue AND opportunity "
    "barriers. An active person with no extra load (high activity, no "
    "burden) faces complacency and reinforcement barriers. An active person "
    "under heavy strain (high activity, major burden) faces scheduling and "
    "fatigue barriers, NOT capability. Every profile has at least one real "
    "barrier; a nudge matched to that barrier is what relieves it.\n\n"
    "CAUSAL RULE: every nudge targets one lever. A nudge that matches the "
    "profile's barrier produces a REAL increase in walking on the day it is "
    "delivered (+150-450 steps; a well-matched nudge targets the middle of "
    "the band, around +300). A nudge that does not match the profile's "
    "barrier produces only a small or negligible increase: someone who "
    "already knows how to walk is not moved by another ability tip, and "
    "someone with no practical obstacle is not moved by an opportunity "
    "reminder.\n\n"
    "EMPIRICAL ANCHOR (PEARL RCT): messages emphasizing ability and "
    "technique were the most effective and best received in the trial (90% "
    "thumbs-up), followed by perceived benefit; planning and prioritization "
    "help specifically when strain (major burden) interferes with acting.\n\n"
    "THEME MAPPING: ability = capability; physical_opportunity = "
    "opportunity (the environment); social_opportunity = opportunity via "
    "other people; perceived_benefit = reflective motivation; planning and "
    "prioritization = self-regulation."
)

_EMPIRICAL_ANCHORS_ACTIONS_OVERRIDES = {
    "idle": "No intervention is delivered today.",
    "ability_morning": (
        "A morning message offers a technique that makes walking easier "
        "(e.g. a shorter route, or a walk paired with your morning coffee); "
        "receiving it, you find walking less effortful and take a noticeably "
        "longer walk this morning. It is a matched nudge for anyone whose "
        "effort or technique is the barrier."
    ),
    "ability_afternoon": (
        "An afternoon message offers a technique that makes walking easier "
        "(e.g. splitting one long walk into two short ones); receiving it, "
        "you find walking less effortful and take a noticeably longer walk "
        "this afternoon. It is a matched nudge for anyone whose effort or "
        "technique is the barrier."
    ),
    "perceived_benefit_morning": (
        "A morning message reminds you what walking does for you (e.g. more "
        "energy and a better mood for the day); it sharpens your motivation, "
        "so you set out more readily and take a noticeably longer walk this "
        "morning. It is a matched nudge for anyone whose motivation or "
        "self-belief is the barrier."
    ),
    "perceived_benefit_afternoon": (
        "An afternoon message reminds you what walking does for you (e.g. a "
        "head-clearing break that helps you sleep); it sharpens your "
        "motivation, so you set out more readily and take a noticeably "
        "longer walk this afternoon. It is a matched nudge for anyone whose "
        "motivation or self-belief is the barrier."
    ),
    "planning_morning": (
        "A morning message helps you schedule today's walk (e.g. a specific "
        "time and route written down before the day fills up); the concrete "
        "plan turns intention into action, so you take a noticeably longer "
        "walk this morning. It is a matched nudge for anyone whose barrier "
        "is scheduling."
    ),
    "planning_afternoon": (
        "An afternoon message helps you schedule the rest of the day (e.g. "
        "blocking out a specific time and route); the concrete plan turns "
        "intention into action, so you take a noticeably longer walk this "
        "afternoon. It is a matched nudge for anyone whose barrier is "
        "scheduling."
    ),
    "prioritization_morning": (
        "A morning message helps you protect time for walking (e.g. moving "
        "a low-priority task off the morning); walking becomes a deliberate "
        "priority, so you take a noticeably longer walk this morning. It is "
        "a matched nudge for anyone whose barrier is time or priority "
        "pressure."
    ),
    "prioritization_afternoon": (
        "An afternoon message helps you protect time for walking (e.g. "
        "delaying a low-priority task to make room); walking becomes a "
        "deliberate priority, so you take a noticeably longer walk this "
        "afternoon. It is a matched nudge for anyone whose barrier is time "
        "or priority pressure."
    ),
    "social_opportunity_morning": (
        "A morning message connects walking to other people (e.g. a friend "
        "who will join you, or a walk-and-talk call); the social opening "
        "gets you out the door, so you take a noticeably longer walk this "
        "morning. It is a matched nudge for anyone whose barrier is a lack "
        "of social support."
    ),
    "social_opportunity_afternoon": (
        "An afternoon message connects walking to other people (e.g. a group "
        "walk or a colleague to join); the social opening gets you out the "
        "door, so you take a noticeably longer walk this afternoon. It is a "
        "matched nudge for anyone whose barrier is a lack of social support."
    ),
    "physical_opportunity_morning": (
        "A morning message points to a physical opening (e.g. favourable "
        "weather or a pleasant nearby route); the practical opportunity "
        "makes the walk easier to take, so you take a noticeably longer walk "
        "this morning. It is a matched nudge for anyone whose barrier is a "
        "lack of practical opportunity."
    ),
    "physical_opportunity_afternoon": (
        "An afternoon message points to a physical opening (e.g. a quieter "
        "path or a stop on the way home); the practical opportunity makes "
        "the walk easier to take, so you take a noticeably longer walk this "
        "afternoon. It is a matched nudge for anyone whose barrier is a lack "
        "of practical opportunity."
    ),
}


def _empirical_anchors_user_extra(state: dict, action: str) -> str:
    """Per-day barrier profile reminder for the empirical_anchors variant."""
    if action == "idle":
        return ""
    profile = _EMPIRICAL_BARRIER_PROFILES.get(
        (state["recent_steps_mean"], state["burden"])
    )
    return (
        "BARRIER PROFILE TODAY: "
        f"this person is {profile or 'facing the barrier described above'}. "
        "A matched nudge adds roughly 150-450 steps on top of their usual "
        "level today; an unmatched nudge adds little."
    )


# Round 5: "protocol" - the full PEARL protocol frame (constitution-alignment
# rung). Replaces round 4's binary matched/unmatched rule with a graded match
# weight per (profile, theme): every nudge day produces a positive response,
# sized by the theme's weight for that day's profile. The idle baseline is
# pinned independent of burden (round 4's high/major idle sat at 6,871, below
# the 7,000 bin, because burden language counteracted the anchor).
_PROTOCOL_SYSTEM_EXTRA = (
    "PROTOCOL FRAME: You are a participant in a year-long adaptive "
    "walking-intervention study. Each day at one decision point the study's "
    "RL system either sends you one of 12 possible nudge messages (a "
    "behavioral theme x time-of-day pair) or no message. Your walking is "
    "tracked every day, and the system learns which messages work for you.\n\n"
    "EMPIRICAL PROTOCOL ANCHORS (PEARL trial): the system favors "
    "ability-improvement messages most often - about 27% of the nudges it "
    "sends, and these get ~90% thumbs-up from participants - with "
    "perceived-benefit and planning messages also frequent. Messages that "
    "improve a barrier you actually have are the ones that raise your "
    "walking that day.\n\n"
    "GRADED MATCH RULE: each nudge theme has a match weight between 0 and 1 "
    "for this person's day. Weight 0.7 or higher -> strong response, in the "
    "middle of the +150-450 step band (around +300). Weight 0.3 to 0.7 -> "
    "modest response (around +120). Weight below 0.3 -> weak but still "
    "slightly positive (around +40). NO intervention-day response is ever "
    "zero or negative: even a poorly matched message beats nothing.\n\n"
    "WEIGHTS BY PROFILE (theme: weight for low/no-burden, low/major-burden, "
    "high/no-burden, high/major-burden):\n"
    "- ability: 0.9, 0.8, 0.5, 0.5 (strong when walking feels hard; moderate "
    "for active people)\n"
    "- perceived_benefit: 0.7, 0.6, 0.7, 0.6 (moderate-to-strong everywhere - "
    "it works through motivation, not logistics)\n"
    "- planning: 0.5, 0.8, 0.4, 0.9 (strong when strain disrupts the "
    "routine)\n"
    "- prioritization: 0.4, 0.7, 0.4, 0.8 (strong when time is tight)\n"
    "- social_opportunity: 0.4, 0.4, 0.5, 0.4 (low-moderate everywhere)\n"
    "- physical_opportunity: 0.5, 0.8, 0.3, 0.8 (strong when the burden is "
    "major)\n\n"
    "IDLE PINNED INDEPENDENT OF BURDEN: on days with no message your steps "
    "stay at your current level regardless of your burden level: around "
    "8,000 if you are a high-activity person, around 3,000 if you are a "
    "low-activity person. Never regress toward the ~5,580 population average."
)

_PROTOCOL_THEME_WEIGHT_CLASS = {
    "ability": ("It is a strong match when walking feels hard, moderate otherwise."),
    "perceived_benefit": (
        "It is a moderate-to-strong match for most profiles (it works "
        "through motivation, not logistics)."
    ),
    "planning": (
        "It is a strong match when strain disrupts the routine, moderate otherwise."
    ),
    "prioritization": ("It is a strong match when time is tight, moderate otherwise."),
    "social_opportunity": ("It is a low-moderate match for most profiles."),
    "physical_opportunity": (
        "It is a strong match when the day's burden is major, low-moderate otherwise."
    ),
}

_PROTOCOL_ACTIONS_OVERRIDES = dict(_COMB_MECHANISMS_ACTION_OVERRIDES)
for _theme, _class_sentence in _PROTOCOL_THEME_WEIGHT_CLASS.items():
    for _time in ("morning", "afternoon"):
        _key = f"{_theme}_{_time}"
        _PROTOCOL_ACTIONS_OVERRIDES[_key] += f" {_class_sentence}"

# (recent_steps_mean, burden) -> profile name + per-theme match weights.
# Weights are spelled out for all 4 profiles x 6 themes so the model never
# has to guess (mirrors the WEIGHTS BY PROFILE table in the system extra).
_PROTOCOL_PROFILE_NAMES = {
    ("low", "none"): "low activity, no burden",
    ("low", "major"): "low activity, major burden",
    ("high", "none"): "high activity, no burden",
    ("high", "major"): "high activity, major burden",
}

_PROTOCOL_PROFILE_WEIGHTS = {
    ("low", "none"): {
        "ability": 0.9,
        "perceived_benefit": 0.7,
        "planning": 0.5,
        "prioritization": 0.4,
        "social_opportunity": 0.4,
        "physical_opportunity": 0.5,
    },
    ("low", "major"): {
        "ability": 0.8,
        "perceived_benefit": 0.6,
        "planning": 0.8,
        "prioritization": 0.7,
        "social_opportunity": 0.4,
        "physical_opportunity": 0.8,
    },
    ("high", "none"): {
        "ability": 0.5,
        "perceived_benefit": 0.7,
        "planning": 0.4,
        "prioritization": 0.4,
        "social_opportunity": 0.5,
        "physical_opportunity": 0.3,
    },
    ("high", "major"): {
        "ability": 0.5,
        "perceived_benefit": 0.6,
        "planning": 0.9,
        "prioritization": 0.8,
        "social_opportunity": 0.4,
        "physical_opportunity": 0.8,
    },
}

_PROTOCOL_THEME_DISPLAY = {
    "ability": "ability",
    "perceived_benefit": "perceived-benefit",
    "planning": "planning",
    "prioritization": "prioritization",
    "social_opportunity": "social-opportunity",
    "physical_opportunity": "physical-opportunity",
}


_STRONG_MATCH_WEIGHT = 0.7
_MODEST_MATCH_WEIGHT = 0.3


def _format_profile_weight_line(
    weight_table: dict[tuple[str, str], dict[str, float]],
    state: dict,
    action: str,
) -> str:
    """Format the per-day profile + theme weight line for one cell.

    Returns an empty string (with a warning) when the profile or theme has
    no configured weight, so missing weights surface in logs instead of
    silently producing an unguided prompt.
    """
    theme = action.rsplit("_", 1)[0]
    key = (state["recent_steps_mean"], state["burden"])
    weight = weight_table.get(key, {}).get(theme)
    if weight is None:
        logger.warning("No configured weight for profile %s / theme %s", key, theme)
        return ""
    if weight >= _STRONG_MATCH_WEIGHT:
        strength = "a strong match"
    elif weight >= _MODEST_MATCH_WEIGHT:
        strength = "a modest match"
    else:
        strength = "a weak match"
    return (
        f"Your profile today: {_PROTOCOL_PROFILE_NAMES[key]}. "
        f"{_PROTOCOL_THEME_DISPLAY[theme]} messages are {strength} (weight "
        f"{weight}) for you today."
    )


def _protocol_user_extra(state: dict, action: str) -> str:
    """Per-day profile + theme weight line for the protocol variant."""
    if action == "idle":
        return ""
    return _format_profile_weight_line(_PROTOCOL_PROFILE_WEIGHTS, state, action)


# Round 6: "protocol_fewshot" - the protocol variant plus prose day-level
# exemplars that calibrate magnitudes (round 5's abstract graded rule
# overshot to +584.9 vs the +150-350 target). The idle pin is re-anchored
# with round 2's +/-500 band trick (7,500-8,500 high / 2,800-3,200 low).
# Exemplars are prose ONLY: JSON-shaped example days would be ingested by
# the response parser as history rows, so the model must never see example
# JSON in that shape.
# Round 7: the never-negative floor is made binding (~+60-100 weak tier),
# the low/no-burden profile gets an explicit 3-barrier breakdown with
# re-weighted themes, and afternoon parity is stated in the causal rule,
# with a low-baseline weak-match exemplar anchoring the floor.
# Round 11 (Option B, literature-backed): exemplars become DELTAS AND
# RANGES ONLY - every absolute step total is removed, because exemplar
# magnitudes leak into outputs as anchors (Min et al. EMNLP 2022; Lou &
# Sun 2025; see docs/research/llm-prompt-calibration-literature.md). The
# only absolute numbers left in the prompt are the idle pin bands, which
# are stated as rules. A JSON-schema grammar block is appended (Wang et
# al. NeurIPS 2023): placeholders keep it unparseable (N/M/A are not
# valid JSON), so it cannot be ingested as a history row.
_PROTOCOL_FEWSHOT_SYSTEM_EXTRA = (
    "PROTOCOL FRAME: You are a participant in a year-long adaptive "
    "walking-intervention study. Each day at one decision point the study's "
    "RL system either sends you one of 12 possible nudge messages (a "
    "behavioral theme x time-of-day pair) or no message. Your walking is "
    "tracked every day, and the system learns which messages work for you.\n\n"
    "EMPIRICAL PROTOCOL ANCHORS (PEARL trial): the system favors "
    "ability-improvement messages most often - about 27% of the nudges it "
    "sends, and these get ~90% thumbs-up from participants - with "
    "perceived-benefit and planning messages also frequent. Messages that "
    "improve a barrier you actually have are the ones that raise your "
    "walking that day.\n\n"
    "GRADED MATCH RULE: each nudge theme has a match weight between 0 and 1 "
    "for this person's day. Weight 0.7 or higher -> strong response, in the "
    "middle of the +150-450 step band (around +300). Weight 0.3 to 0.7 -> "
    "modest response (~+120 to +180). Weight below 0.3 -> small but clearly "
    "positive (~+60 to +100). A message NEVER reduces your steps, no matter "
    "how poorly it matches. Even the weakest match raises your day's total "
    "by a small amount. This holds for morning AND afternoon messages "
    "alike. An afternoon message is just as likely to raise your steps as a "
    "morning message; the time of day only changes which part of the day "
    "the extra steps land in. The +150-450 step band is also a ceiling: no "
    "message raises a day's total by more than about 500 steps, no matter "
    "how well it matches.\n\n"
    "WEIGHTS BY PROFILE (theme: weight for low/no-burden, low/major-burden, "
    "high/no-burden, high/major-burden):\n"
    "- ability: 0.8, 0.8, 0.5, 0.5 (strong when walking feels hard; moderate "
    "for active people)\n"
    "- perceived_benefit: 0.8, 0.6, 0.7, 0.6 (moderate-to-strong everywhere - "
    "it works through motivation, not logistics)\n"
    "- planning: 0.7, 0.8, 0.4, 0.9 (strong when strain disrupts the "
    "routine or when no walking routine exists yet)\n"
    "- prioritization: 0.4, 0.7, 0.4, 0.8 (strong when time is tight)\n"
    "- social_opportunity: 0.3, 0.4, 0.5, 0.4 (low-moderate everywhere)\n"
    "- physical_opportunity: 0.5, 0.8, 0.3, 0.8 (strong when the burden is "
    "major)\n\n"
    "LOW NO-BURDEN PROFILE: a struggling walker with no extra load has "
    "three real barriers, each relieved by a different theme. (a) Effort "
    "and technique: walking feels hard and they doubt they can do it, so "
    "ability messages are a strong match (weight 0.8). (b) Motivation and "
    "energy dips: they run out of steam through the day, so "
    "perceived-benefit messages are a strong match (weight 0.8 - they work "
    "through motivation, not logistics). (c) Lack of routine: there is no "
    "daily walking habit to fall back on, so planning messages are a good "
    "match (weight 0.7) for building one. Prioritization (0.4) and "
    "physical_opportunity (0.5) are modest matches; social_opportunity "
    "(0.3) is the weakest.\n\n"
    "IDLE PINNED INDEPENDENT OF BURDEN: on days with no message your steps "
    "stay at your current level regardless of your burden level: between "
    "7,500 and 8,500 if you are a high-activity person, between 2,800 and "
    "3,200 if you are a low-activity person. Never regress toward the ~5,580 "
    "population average.\n\n"
    "DAY-LEVEL EXEMPLARS (increments, not day totals - the size of the "
    "response is what matters, never the absolute numbers): A strongly "
    "matched message (weight >= 0.7) raises the day's total by about 250 "
    "to 350 steps above the person's no-message day. A modestly matched "
    "message (weight ~0.4) raises the total by about 120 to 180 steps. A "
    "weakly matched message (weight ~0.3) raises the total by about 60 to "
    "100 steps - small, but clearly present. The same increment applies "
    "for a low-activity person as for a high-activity person: the message "
    "adds steps on top of their own normal day, the extra steps land "
    "mostly in the half of the day that received the message, and their "
    "normal day itself stays inside their pinned idle band.\n\n"
    "OUTPUT FORMAT: respond with exactly 7 lines, one JSON object per "
    'line, of the form {"day": N, "morning_steps": M, "afternoon_steps": '
    "A} with N = 1, 2, ..., 7 and M, A plain integers (no quotes, no "
    "thousands separators). No reasoning, no markdown fences, no "
    "commentary, no other text."
)

_PROTOCOL_FEWSHOT_ACTIONS_OVERRIDES = dict(_PROTOCOL_ACTIONS_OVERRIDES)

# Round 5's profile + weight line is kept, but reading the fewshot weight
# table: round 7 re-weights the low/no-burden profile (ability 0.8,
# perceived_benefit 0.8, planning 0.7, prioritization 0.4,
# physical_opportunity 0.5, social_opportunity 0.3) so the day-level line
# stays consistent with the WEIGHTS BY PROFILE table above. The other
# three profiles keep the round-5 weights.
_PROTOCOL_FEWSHOT_PROFILE_WEIGHTS = dict(_PROTOCOL_PROFILE_WEIGHTS)
_PROTOCOL_FEWSHOT_PROFILE_WEIGHTS[("low", "none")] = {
    "ability": 0.8,
    "perceived_benefit": 0.8,
    "planning": 0.7,
    "prioritization": 0.4,
    "social_opportunity": 0.3,
    "physical_opportunity": 0.5,
}


def _protocol_fewshot_user_extra(state: dict, action: str) -> str:
    """Per-day profile + theme weight line for the fewshot variant."""
    if action == "idle":
        return ""
    return _format_profile_weight_line(_PROTOCOL_FEWSHOT_PROFILE_WEIGHTS, state, action)


_PROTOCOL_FEWSHOT_USER_EXTRA = _protocol_fewshot_user_extra


PROMPT_VARIANT_CONFIGS: dict[str, PromptVariant] = {
    "baseline": PromptVariant(),
    "state_self_model": PromptVariant(
        system_extra=_SELF_MODEL_SYSTEM_EXTRA,
        user_extra=_state_self_model_user_extra,
    ),
    "com_b_mechanisms": PromptVariant(
        system_extra=_COMB_MECHANISMS_SYSTEM_EXTRA,
        action_overrides=_COMB_MECHANISMS_ACTION_OVERRIDES,
        user_extra=_COMB_MECHANISMS_USER_EXTRA,
    ),
    "empirical_anchors": PromptVariant(
        system_extra=_EMPIRICAL_ANCHORS_SYSTEM_EXTRA,
        action_overrides=_EMPIRICAL_ANCHORS_ACTIONS_OVERRIDES,
        user_extra=_empirical_anchors_user_extra,
    ),
    "protocol": PromptVariant(
        system_extra=_PROTOCOL_SYSTEM_EXTRA,
        action_overrides=_PROTOCOL_ACTIONS_OVERRIDES,
        user_extra=_protocol_user_extra,
    ),
    "protocol_fewshot": PromptVariant(
        system_extra=_PROTOCOL_FEWSHOT_SYSTEM_EXTRA,
        action_overrides=_PROTOCOL_FEWSHOT_ACTIONS_OVERRIDES,
        user_extra=_PROTOCOL_FEWSHOT_USER_EXTRA,
    ),
}

# Single source of truth for variant names (also used for CLI validation).
PROMPT_VARIANTS = tuple(PROMPT_VARIANT_CONFIGS)

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
