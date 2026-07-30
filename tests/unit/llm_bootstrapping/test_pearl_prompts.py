"""Tests for llm_bootstrapping.prompts.pearl module."""

from __future__ import annotations

from rl_health_interventions.llm_bootstrapping.prompts.pearl import (
    ACTIONS,
    BURDENS,
    DAY_TYPES,
    MORNING_RATIOS,
    RECENT_STEPS_MEAN,
    WALK_PATTERNS,
    _render_system_prompt,
    _render_user_prompt,
    generate_prompts,
)


class TestRenderSystemPrompt:
    def test_contains_persona_description(self) -> None:
        prompt = _render_system_prompt("base")
        assert "persona" in prompt.lower() or "person" in prompt.lower()

    def test_contains_baseline_steps(self) -> None:
        prompt = _render_system_prompt("base")
        assert "5,580" in prompt or "5580" in prompt

    def test_contains_effect_size(self) -> None:
        prompt = _render_system_prompt("base")
        assert "150" in prompt
        assert "450" in prompt

    def test_all_personas(self) -> None:
        for persona in (
            "base",
            "goal_driven",
            "social_responder",
            "stable_maintainer",
            "resistant",
        ):
            prompt = _render_system_prompt(persona)
            assert len(prompt) > 100


class TestRenderUserPrompt:
    def test_contains_state_description(self) -> None:
        prompt = _render_user_prompt(
            recent_steps_mean="low",
            walk_pattern="low",
            morning_ratio="morning",
            day_type="weekday",
            burden="none",
            action="idle",
        )
        assert "low" in prompt.lower()
        assert "morning" in prompt.lower()

    def test_contains_action_description(self) -> None:
        prompt = _render_user_prompt(
            recent_steps_mean="moderate",
            walk_pattern="high",
            morning_ratio="balanced",
            day_type="weekend",
            burden="minor",
            action="ability_morning",
        )
        assert "ability" in prompt.lower() or "morning" in prompt.lower()

    def test_contains_task_instruction(self) -> None:
        prompt = _render_user_prompt(
            recent_steps_mean="high",
            walk_pattern="high",
            morning_ratio="evening",
            day_type="weekday",
            burden="major",
            action="idle",
        )
        assert "7 days" in prompt
        assert "morning_steps" in prompt
        assert "afternoon_steps" in prompt

    def test_output_format_specified(self) -> None:
        prompt = _render_user_prompt(
            recent_steps_mean="low",
            walk_pattern="low",
            morning_ratio="morning",
            day_type="weekday",
            burden="none",
            action="idle",
        )
        assert '{"day"' in prompt


class TestGeneratePrompts:
    def test_returns_system_and_list(self) -> None:
        system, prompts = generate_prompts(persona="base")
        assert isinstance(system, str)
        assert isinstance(prompts, list)
        assert len(system) > 0

    def test_with_state_subset(self) -> None:
        states = [
            {
                "recent_steps_mean": "low",
                "recent_walk_pattern": "low",
                "morning_steps_ratio": "morning",
                "day_of_week": "weekday",
                "burden": "none",
            },
        ]
        _system, prompts = generate_prompts(
            persona="base", samples_per_cell=2, state_subset=states
        )
        # 1 state x 13 actions x 2 samples = 26
        assert len(prompts) == 26

    def test_full_generation_count(self) -> None:
        # 3x2x3x2x3 = 108 states including burden
        # 108 states x 13 actions x 10 samples = 14,040
        _, prompts = generate_prompts(persona="base", samples_per_cell=10)
        assert len(prompts) == 14040

    def test_samples_per_cell_repeats(self) -> None:
        states = [
            {
                "recent_steps_mean": "low",
                "recent_walk_pattern": "low",
                "morning_steps_ratio": "morning",
                "day_of_week": "weekday",
                "burden": "none",
            },
        ]
        _, prompts = generate_prompts(
            persona="base", samples_per_cell=3, state_subset=states
        )
        # Same prompt repeated 3 times per action
        # Check first action's prompts are identical
        first_3 = prompts[:3]
        assert first_3[0] == first_3[1] == first_3[2]

    def test_all_actions_covered(self) -> None:
        from rl_health_interventions.llm_bootstrapping.prompts.pearl import (
            ACTION_DESCRIPTIONS,
        )

        states = [
            {
                "recent_steps_mean": "low",
                "recent_walk_pattern": "low",
                "morning_steps_ratio": "morning",
                "day_of_week": "weekday",
                "burden": "none",
            },
        ]
        _, prompts = generate_prompts(
            persona="base", samples_per_cell=1, state_subset=states
        )
        # Each action description should appear in prompts
        for action, desc in ACTION_DESCRIPTIONS.items():
            # Check for a key phrase from the description
            key_phrase = desc.split("(")[0].strip()[:30]
            found = any(key_phrase.lower() in p.lower() for p in prompts)
            assert found, f"Action {action} description not found in prompts"

    def test_state_factors_covered(self) -> None:
        # 3x2x3x2x3 = 108 states (all factors including burden)
        states = [
            {
                "recent_steps_mean": rsm,
                "recent_walk_pattern": wp,
                "morning_steps_ratio": mr,
                "day_of_week": dt,
                "burden": b,
            }
            for rsm in RECENT_STEPS_MEAN
            for wp in WALK_PATTERNS
            for mr in MORNING_RATIOS
            for dt in DAY_TYPES
            for b in BURDENS
        ]
        _, prompts = generate_prompts(
            persona="base", samples_per_cell=1, state_subset=states
        )
        assert len(prompts) == 108 * 13  # 1,404


class TestConstants:
    def test_actions_count(self) -> None:
        assert len(ACTIONS) == 13  # idle + 12 COM-B x time

    def test_burdens_match_config(self) -> None:
        assert BURDENS == ["none", "minor", "major"]

    def test_recent_steps_mean_levels(self) -> None:
        assert RECENT_STEPS_MEAN == ["low", "moderate", "high"]

    def test_walk_patterns(self) -> None:
        assert WALK_PATTERNS == ["low", "high"]

    def test_morning_ratios(self) -> None:
        assert MORNING_RATIOS == ["morning", "balanced", "evening"]

    def test_day_types(self) -> None:
        assert DAY_TYPES == ["weekday", "weekend"]
