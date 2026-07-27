from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
from typing_extensions import override

from rl_health_interventions.agents._base import Agent

_THEMES = frozenset(
    {
        "ability",
        "perceived_benefit",
        "planning",
        "prioritization",
        "social_opportunity",
        "physical_opportunity",
    }
)
_TIMINGS = frozenset({"morning", "afternoon"})
_MAX_LIKERT = 5
_VALID_TIME_PREFERENCES = frozenset({"morning", "afternoon", "no_preference"})

_ALL_ACTIONS = sorted(
    f"{theme}_{timing}" for theme in sorted(_THEMES) for timing in sorted(_TIMINGS)
)


class FixedAgent(Agent):
    def __init__(
        self,
        action: str = "idle",
        seed: int | None = None,  # noqa: ARG002
        actions: list[str] | None = None,  # noqa: ARG002
    ) -> None:
        self._action = action

    @override
    def select_action(self, state) -> str:
        return self._action


def _validate_time_preference(time_pref: str, persona_name: str, path: str) -> None:
    """Raise ValueError if time_pref is not a valid value."""
    if time_pref not in _VALID_TIME_PREFERENCES:
        raise ValueError(
            f"invalid time_preference '{time_pref}' for persona '{persona_name}' "
            f"in {path}; expected one of {sorted(_VALID_TIME_PREFERENCES)}"
        )


def _load_scores_from_file(
    path: str,
    persona_name: str | None,
    time_preference: str | None,
) -> tuple[dict[str, int], str]:
    """Load COM-B scores and time_preference for a persona from a JSON file."""
    if persona_name is None:
        raise ValueError(
            "persona_name must be provided when persona_comb_file is given"
        )
    filepath = pathlib.Path(path).resolve()
    with filepath.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    if persona_name not in data:
        raise ValueError(
            f"persona '{persona_name}' not found in {path}; "
            f"available: {sorted(data.keys())}"
        )
    entry = data[persona_name]
    scores = {}
    for theme in sorted(_THEMES):
        if theme not in entry:
            raise ValueError(
                f"theme '{theme}' missing for persona '{persona_name}' in {path}"
            )
        scores[theme] = entry[theme]
    time_pref = time_preference or entry.get("time_preference", "no_preference")
    _validate_time_preference(time_pref, persona_name, path)
    return scores, time_pref


class ComBWeightedFixedAgent(Agent):
    THEMES = _THEMES
    TIMINGS = _TIMINGS

    def __init__(
        self,
        actions: list[str] | None = None,
        seed: int = 42,
        comb_scores: dict[str, int] | None = None,
        persona_comb_file: str | None = None,
        persona_name: str | None = None,
        time_preference: str | None = None,
    ) -> None:
        self._actions = self._resolve_actions(actions)
        scores, time_preference = self._resolve_scores(
            comb_scores, persona_comb_file, persona_name, time_preference
        )
        self._time_preference = time_preference or "no_preference"
        self._validate_scores(scores)
        self._setup_barriers(scores)
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _resolve_actions(actions: list[str] | None) -> list[str]:
        """Validate and return the action list."""
        if actions is None:
            return list(_ALL_ACTIONS)
        result = list(actions)
        expected = set(_ALL_ACTIONS)
        missing = expected - set(result)
        if missing:
            raise ValueError(
                f"actions must contain all 12 {{theme}}_{{timing}} combinations; "
                f"missing: {sorted(missing)}"
            )
        return result

    @staticmethod
    def _resolve_scores(
        comb_scores: dict[str, int] | None,
        persona_comb_file: str | None,
        persona_name: str | None,
        time_preference: str | None,
    ) -> tuple[dict[str, int], str]:
        """Load COM-B scores from inline dict or JSON file."""
        has_both = comb_scores is not None and persona_comb_file is not None
        has_none = comb_scores is None and persona_comb_file is None
        if has_both:
            raise ValueError("comb_scores and persona_comb_file are mutually exclusive")
        if has_none:
            raise ValueError("either comb_scores or persona_comb_file must be provided")

        if persona_comb_file is not None:
            return _load_scores_from_file(
                persona_comb_file, persona_name, time_preference
            )

        raw = dict(comb_scores) if comb_scores is not None else {}
        return raw, (time_preference or "no_preference")

    def _setup_barriers(self, scores: dict[str, int]) -> None:
        """Compute and store barrier values from COM-B scores."""
        barriers: dict[str, int] = {}
        for theme in sorted(_THEMES):
            barrier = _MAX_LIKERT - scores[theme]
            barriers[theme] = barrier

        non_zero = {t: b for t, b in barriers.items() if b > 0}
        if not non_zero:
            raise ValueError(
                "all COM-B barriers are zero (all themes scored 5); "
                "no action has non-zero probability"
            )

        self._barriers = barriers
        self._non_zero_themes = sorted(non_zero.keys())
        self._non_zero_weights = [float(non_zero[t]) for t in self._non_zero_themes]

    @staticmethod
    def _validate_scores(scores: dict[str, int]) -> None:
        if not scores:
            raise ValueError("comb_scores must be a non-empty dict")
        for theme in sorted(_THEMES):
            if theme not in scores:
                raise ValueError(
                    f"theme '{theme}' missing from comb_scores; "
                    f"required: {sorted(_THEMES)}"
                )
            val = scores[theme]
            if not isinstance(val, int) or val < 1 or val > _MAX_LIKERT:
                raise ValueError(
                    f"comb_scores['{theme}'] must be an int in [1, {_MAX_LIKERT}], "
                    f"got {val!r}"
                )

    @override
    def select_action(self, state: Any) -> str:
        # Sample theme from multinomial weighted by barriers
        theme = self._rng.choice(
            self._non_zero_themes,
            p=self._non_zero_weights / np.sum(self._non_zero_weights),
        )

        # Sample timing based on preference
        if self._time_preference == "morning":
            morning_p = 0.7
        elif self._time_preference == "afternoon":
            morning_p = 0.3
        else:  # no_preference
            morning_p = 0.5

        timing = self._rng.choice(
            ["morning", "afternoon"], p=[morning_p, 1.0 - morning_p]
        )
        return f"{theme}_{timing}"

    @override
    def update(
        self,
        state: Any,
        action: str,
        reward: float,
        next_state: Any,
        done: bool = False,
    ) -> None:
        return None

    @override
    def on_day_end(self) -> None:
        return None


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY.register("fixed", FixedAgent)
    REGISTRY.register("comb_weighted_fixed", ComBWeightedFixedAgent)
