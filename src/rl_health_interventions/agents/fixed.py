from __future__ import annotations

import json
import random
from pathlib import Path

from typing_extensions import override

from rl_health_interventions.agents._base import Agent


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


class ComBWeightedFixedAgent(Agent):
    _THEMES = (
        "ability",
        "perceived_benefit",
        "planning",
        "prioritization",
        "social_opportunity",
        "physical_opportunity",
    )

    def __init__(
        self,
        comb_scores: dict[str, int] | None = None,
        persona_comb_file: str | None = None,
        persona_name: str | None = None,
        time_preference: str = "no_preference",
        seed: int | None = None,
        actions: list[str] | None = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._actions = set(actions or [])
        self._comb_scores = self._resolve_scores(comb_scores, persona_comb_file, persona_name)
        self._time_preference = time_preference

    def _resolve_scores(
        self,
        comb_scores: dict[str, int] | None,
        persona_comb_file: str | None,
        persona_name: str | None,
    ) -> dict[str, int]:
        if comb_scores is not None:
            return {theme: int(comb_scores.get(theme, 3)) for theme in self._THEMES}

        if persona_comb_file is None:
            return {theme: 3 for theme in self._THEMES}

        if persona_name is None or not persona_name.strip():
            raise ValueError("persona_name must be provided with persona_comb_file")

        with Path(persona_comb_file).open(encoding="utf-8") as f:
            data = json.load(f)
        if persona_name not in data:
            raise ValueError(f"persona '{persona_name}' not found in {persona_comb_file}")
        persona_data = data[persona_name]
        return {theme: int(persona_data.get(theme, 3)) for theme in self._THEMES}

    def _sample_theme(self) -> str:
        barriers = [max(0, 5 - self._comb_scores.get(theme, 3)) for theme in self._THEMES]
        if sum(barriers) == 0:
            barriers = [1] * len(self._THEMES)
        return self._rng.choices(list(self._THEMES), weights=barriers, k=1)[0]

    def _sample_timing(self) -> str:
        if self._time_preference == "morning":
            weights = (0.7, 0.3)
        elif self._time_preference == "afternoon":
            weights = (0.3, 0.7)
        else:
            weights = (0.5, 0.5)
        return self._rng.choices(["morning", "afternoon"], weights=weights, k=1)[0]

    def _fallback_action(self) -> str:
        non_idle = sorted(a for a in self._actions if a != "idle")
        if non_idle:
            return non_idle[0]
        return "idle"

    @override
    def select_action(self, state) -> str:  # noqa: ARG002
        action = f"{self._sample_theme()}_{self._sample_timing()}"
        if self._actions and action not in self._actions:
            return self._fallback_action()
        return action


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY.register("fixed", FixedAgent)
    REGISTRY.register("comb_weighted_fixed", ComBWeightedFixedAgent)
