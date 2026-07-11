"""Tests for llm_bootstrapping.prompts.sprint1 module."""

from __future__ import annotations

import pytest

from rl_health_interventions.llm_bootstrapping.prompts import (
    generate_prompts,
)


class TestGeneratePrompts:
    def test_returns_system_prompt_and_prompts(self) -> None:
        system, prompts = generate_prompts(persona="base")
        assert isinstance(system, str)
        assert isinstance(prompts, list)
        assert len(system) > 0
        assert len(prompts) > 0

    def test_default_count(self) -> None:
        _, prompts = generate_prompts(persona="base")
        assert len(prompts) == 22320

    def test_unknown_persona(self) -> None:
        with pytest.raises(ValueError, match="Unknown persona"):
            generate_prompts(persona="nonexistent")

    def test_all_personas_produce_system_prompt(self) -> None:
        for persona in (
            "base",
            "goal_driven",
            "social_responder",
            "resistant",
            "stable_maintainer",
        ):
            system, _ = generate_prompts(persona=persona)
            assert len(system) > 0

    def test_prompts_contain_expected_text(self) -> None:
        _, prompts = generate_prompts(persona="base")
        first = prompts[0]
        assert "# Current state" in first
        assert "day_type" in first or "steps" in first or "sleep" in first

    def test_day_boundary_prompts_first(self) -> None:
        _, prompts = generate_prompts(persona="base")
        first_720 = prompts[:720]
        for p in first_720:
            assert "How many steps" not in p

    def test_within_day_prompts_have_step_query(self) -> None:
        _, prompts = generate_prompts(persona="base")
        within_start = 720
        within = prompts[within_start]
        assert "How many steps" in within

    def test_correct_day_boundary_count(self) -> None:
        _, prompts = generate_prompts(persona="base", samples_per_cell=10)
        db_count = sum(1 for p in prompts if "It's bedtime" in p)
        assert db_count == 720

    def test_within_day_day_boundary_separator(self) -> None:
        _, prompts = generate_prompts(persona="base")
        transitions = [p for p in prompts if "last night was" in p]
        assert len(transitions) == 720

    def test_system_prompt_contains_reference(self) -> None:
        system, _ = generate_prompts(persona="base")
        assert "Reference" in system
        assert "inactive" in system
        assert "moderate" in system
        assert "active" in system
