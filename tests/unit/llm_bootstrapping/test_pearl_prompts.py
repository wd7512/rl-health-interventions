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
            found = any(key_phrase.lower() in p[0].lower() for p in prompts)
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


class TestStateSelfModelVariant:
    """State-conditional self-model anchors (round 2 variant)."""

    def test_system_extra_reframes_population_average(self) -> None:
        prompt = _render_system_prompt("base", prompt_variant="state_self_model")
        assert "5,580" in prompt
        assert "population average" in prompt
        assert "never regress" in prompt

    def test_user_extra_is_state_conditional(self) -> None:
        high_prompt = _render_user_prompt(
            recent_steps_mean="high",
            walk_pattern="high",
            morning_ratio="balanced",
            day_type="weekday",
            burden="none",
            action="idle",
            prompt_variant="state_self_model",
        )
        low_prompt = _render_user_prompt(
            recent_steps_mean="low",
            walk_pattern="low",
            morning_ratio="balanced",
            day_type="weekday",
            burden="none",
            action="idle",
            prompt_variant="state_self_model",
        )
        assert "8,000" in high_prompt
        assert "3,000" in low_prompt
        assert "8,000" not in low_prompt

    def test_idle_vs_intervention_day_lines(self) -> None:
        idle_prompt = _render_user_prompt(
            recent_steps_mean="moderate",
            walk_pattern="low",
            morning_ratio="balanced",
            day_type="weekday",
            burden="none",
            action="idle",
            prompt_variant="state_self_model",
        )
        ability_prompt = _render_user_prompt(
            recent_steps_mean="moderate",
            walk_pattern="low",
            morning_ratio="balanced",
            day_type="weekday",
            burden="none",
            action="ability_morning",
            prompt_variant="state_self_model",
        )
        assert "no step boost" in idle_prompt
        assert "150-450 steps" in ability_prompt

    def test_extra_before_json_instruction(self) -> None:
        prompt = _render_user_prompt(
            recent_steps_mean="low",
            walk_pattern="low",
            morning_ratio="morning",
            day_type="weekday",
            burden="none",
            action="idle",
            prompt_variant="state_self_model",
        )
        assert prompt.index("SELF-MODEL ANCHORS") < prompt.index(
            "Output exactly 7 JSON objects"
        )

    def test_extra_contains_no_json_objects(self) -> None:
        prompt = _render_user_prompt(
            recent_steps_mean="low",
            walk_pattern="low",
            morning_ratio="morning",
            day_type="weekday",
            burden="none",
            action="idle",
            prompt_variant="state_self_model",
        )
        assert '{"day"' in prompt
        assert prompt.count('{"day"') == 3  # only the 3 format examples


class TestCombMechanismsVariant:
    """COM-B causal mechanism framing (round 3 variant)."""

    def test_system_extra_explains_com_b_and_causal_rule(self) -> None:
        prompt = _render_system_prompt("base", prompt_variant="com_b_mechanisms")
        assert "COM-B" in prompt
        assert "Capability" in prompt
        assert "Opportunity" in prompt
        assert "Motivation" in prompt
        assert "self-regulation" in prompt
        assert "+150-450 steps" in prompt
        assert "negligible" in prompt

    def test_theme_mapping_present(self) -> None:
        prompt = _render_system_prompt("base", prompt_variant="com_b_mechanisms")
        assert "ability = capability" in prompt
        assert "physical_opportunity = opportunity" in prompt
        assert "social_opportunity = opportunity via other people" in prompt
        assert "perceived_benefit = reflective motivation" in prompt

    def test_persistence_rule_kept(self) -> None:
        prompt = _render_system_prompt("base", prompt_variant="com_b_mechanisms")
        assert "never regress" in prompt

    def test_all_actions_overridden_with_mechanism(self) -> None:
        for action in ACTIONS:
            prompt = _render_user_prompt(
                recent_steps_mean="moderate",
                walk_pattern="low",
                morning_ratio="balanced",
                day_type="weekday",
                burden="none",
                action=action,
                prompt_variant="com_b_mechanisms",
            )
            assert "nudge" in prompt
            if action == "idle":
                assert "No intervention is delivered today." in prompt
            else:
                assert "noticeably longer walk" in prompt

    def test_morning_afternoon_mechanism_sentences_distinct(self) -> None:
        for theme in (
            "ability",
            "perceived_benefit",
            "planning",
            "prioritization",
            "social_opportunity",
            "physical_opportunity",
        ):
            morning = _render_user_prompt(
                recent_steps_mean="moderate",
                walk_pattern="low",
                morning_ratio="balanced",
                day_type="weekday",
                burden="none",
                action=f"{theme}_morning",
                prompt_variant="com_b_mechanisms",
            )
            afternoon = _render_user_prompt(
                recent_steps_mean="moderate",
                walk_pattern="low",
                morning_ratio="balanced",
                day_type="weekday",
                burden="none",
                action=f"{theme}_afternoon",
                prompt_variant="com_b_mechanisms",
            )
            assert morning != afternoon

    def test_user_extra_closing_line(self) -> None:
        prompt = _render_user_prompt(
            recent_steps_mean="low",
            walk_pattern="low",
            morning_ratio="morning",
            day_type="weekday",
            burden="none",
            action="ability_morning",
            prompt_variant="com_b_mechanisms",
        )
        assert "matched nudge produces a clear increase" in prompt
        assert "unmatched or no nudge leaves your steps near your usual level" in prompt

    def test_system_extra_before_json_instruction(self) -> None:
        system = _render_system_prompt("base", prompt_variant="com_b_mechanisms")
        assert "COM-B MECHANISMS" in system

    def test_user_extra_before_json_instruction(self) -> None:
        prompt = _render_user_prompt(
            recent_steps_mean="low",
            walk_pattern="low",
            morning_ratio="morning",
            day_type="weekday",
            burden="none",
            action="ability_morning",
            prompt_variant="com_b_mechanisms",
        )
        assert prompt.index("matched nudge produces a clear increase") < prompt.index(
            "Output exactly 7 JSON objects"
        )

    def test_extra_contains_no_json_objects(self) -> None:
        prompt = _render_user_prompt(
            recent_steps_mean="low",
            walk_pattern="low",
            morning_ratio="morning",
            day_type="weekday",
            burden="none",
            action="ability_morning",
            prompt_variant="com_b_mechanisms",
        )
        assert '{"day"' in prompt
        assert prompt.count('{"day"') == 3  # only the 3 format examples


class TestEmpiricalAnchorsVariant:
    """Numeric anchors + barrier profiles + mechanisms (round 4 variant)."""

    def test_system_extra_has_numeric_anchors(self) -> None:
        prompt = _render_system_prompt("base", prompt_variant="empirical_anchors")
        assert "population average" in prompt
        assert "under 4,000" in prompt
        assert "4,000-7,000" in prompt
        assert "over 7,000" in prompt
        assert "never regress" in prompt
        assert "around 8,000" in prompt
        assert "around 3,000" in prompt

    def test_system_extra_has_barrier_profiles(self) -> None:
        prompt = _render_system_prompt("base", prompt_variant="empirical_anchors")
        assert "BARRIER PROFILE" in prompt
        assert "complacency" in prompt
        assert "fatigue AND opportunity" in prompt
        assert "NOT capability" in prompt
        assert "Every profile has at least one real barrier" in prompt

    def test_system_extra_has_causal_rule_and_empirical_anchor(self) -> None:
        prompt = _render_system_prompt("base", prompt_variant="empirical_anchors")
        assert "+150-450 steps" in prompt
        assert "middle of the band" in prompt
        assert "90%" in prompt
        assert "thumbs-up" in prompt
        assert "planning and prioritization" in prompt
        assert "negligible" in prompt

    def test_all_actions_overridden(self) -> None:
        for action in ACTIONS:
            prompt = _render_user_prompt(
                recent_steps_mean="moderate",
                walk_pattern="low",
                morning_ratio="balanced",
                day_type="weekday",
                burden="none",
                action=action,
                prompt_variant="empirical_anchors",
            )
            if action == "idle":
                assert "No intervention is delivered today." in prompt
                continue
            assert "nudge" in prompt
            assert "noticeably longer walk" in prompt
            assert "matched nudge" in prompt

    def test_matched_clauses_per_theme(self) -> None:
        expected = {
            "ability": "effort or technique",
            "perceived_benefit": "motivation or self-belief",
            "planning": "barrier is scheduling",
            "prioritization": "time or priority pressure",
            "social_opportunity": "social support",
            "physical_opportunity": "practical opportunity",
        }
        for theme, clause in expected.items():
            prompt = _render_user_prompt(
                recent_steps_mean="low",
                walk_pattern="low",
                morning_ratio="balanced",
                day_type="weekday",
                burden="major",
                action=f"{theme}_morning",
                prompt_variant="empirical_anchors",
            )
            assert clause in prompt

    def test_user_extra_empty_on_idle(self) -> None:
        prompt = _render_user_prompt(
            recent_steps_mean="high",
            walk_pattern="high",
            morning_ratio="balanced",
            day_type="weekday",
            burden="none",
            action="idle",
            prompt_variant="empirical_anchors",
        )
        assert "BARRIER PROFILE TODAY" not in prompt
        assert '{"day"' in prompt

    def test_user_extra_profile_is_state_conditional(self) -> None:
        high_none = _render_user_prompt(
            recent_steps_mean="high",
            walk_pattern="low",
            morning_ratio="balanced",
            day_type="weekday",
            burden="none",
            action="ability_morning",
            prompt_variant="empirical_anchors",
        )
        low_major = _render_user_prompt(
            recent_steps_mean="low",
            walk_pattern="low",
            morning_ratio="balanced",
            day_type="weekday",
            burden="major",
            action="ability_morning",
            prompt_variant="empirical_anchors",
        )
        assert "complacency" in high_none
        assert "complacency" not in low_major
        assert "fatigue AND opportunity" in low_major
        assert "150-450 steps" in high_none
        assert "150-450 steps" in low_major

    def test_extra_before_json_instruction(self) -> None:
        prompt = _render_user_prompt(
            recent_steps_mean="low",
            walk_pattern="low",
            morning_ratio="morning",
            day_type="weekday",
            burden="none",
            action="ability_morning",
            prompt_variant="empirical_anchors",
        )
        assert prompt.index("BARRIER PROFILE TODAY") < prompt.index(
            "Output exactly 7 JSON objects"
        )

    def test_extra_contains_no_json_objects(self) -> None:
        prompt = _render_user_prompt(
            recent_steps_mean="low",
            walk_pattern="low",
            morning_ratio="morning",
            day_type="weekday",
            burden="none",
            action="ability_morning",
            prompt_variant="empirical_anchors",
        )
        assert '{"day"' in prompt
        assert prompt.count('{"day"') == 3  # only the 3 format examples


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
