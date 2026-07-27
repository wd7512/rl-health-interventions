import json

import pytest

from rl_health_interventions.agents.fixed import ComBWeightedFixedAgent, FixedAgent
from rl_health_interventions.config.schemas import AgentConfig


def test_fixed_agent_returns_configured_action():
    agent = FixedAgent(action="nudge")
    result = agent.select_action(None)
    assert result == "nudge"


def test_fixed_agent_defaults_to_idle():
    agent = FixedAgent()
    assert agent.select_action(None) == "idle"


def test_fixed_agent_ignores_state():
    agent = FixedAgent(action="idle")
    assert agent.select_action("any_state") == "idle"


def test_fixed_agent_update_is_noop():
    agent = FixedAgent(action="nudge")
    agent.update(None, "nudge", 1.0, None)
    assert agent.select_action(None) == "nudge"


def test_fixed_agent_requires_action():
    with pytest.raises(ValueError, match="action must be provided"):
        AgentConfig.model_validate({"type": "fixed"})


@pytest.mark.parametrize("action", ["", "  ", 123])
def test_fixed_agent_rejects_invalid_action(action):
    with pytest.raises((ValueError, TypeError)):
        AgentConfig.model_validate({"type": "fixed", "action": action})


@pytest.mark.parametrize(
    "field",
    [
        {"alpha_prior": 1.0},
        {"beta_prior": 1.0},
        {"epsilon": 0.1},
        {"epsilon_start": 0.2},
        {"c": 2.0},
        {"contextual": True},
    ],
)
def test_fixed_agent_rejects_learning_params(field):
    with pytest.raises(ValueError, match="does not accept"):
        AgentConfig.model_validate({"type": "fixed", "action": "nudge", **field})


# ──────────────────────────────────────────────────────────────────────
# ComBWeightedFixedAgent
# ──────────────────────────────────────────────────────────────────────

_INLINE_SCORES = {
    "ability": 3,
    "perceived_benefit": 2,
    "physical_opportunity": 4,
    "planning": 2,
    "prioritization": 3,
    "social_opportunity": 3,
}


class TestComBWeightedFixedAgent:
    def test_single_dominant_barrier(self):
        """Single dominant barrier leads to 100% that theme."""
        scores = {
            "ability": 1,  # barrier = 4
            "perceived_benefit": 5,  # barrier = 0
            "physical_opportunity": 5,  # barrier = 0
            "planning": 5,  # barrier = 0
            "prioritization": 5,  # barrier = 0
            "social_opportunity": 5,  # barrier = 0
        }
        agent = ComBWeightedFixedAgent(comb_scores=scores, seed=42)
        actions = [agent.select_action(None) for _ in range(1000)]
        themes = {a.rsplit("_", 1)[0] for a in actions}
        assert themes == {"ability"}

    def test_equal_barriers_uniform(self):
        """All equal barriers → uniform theme distribution."""
        scores = {
            "ability": 3,  # barrier = 2
            "perceived_benefit": 3,  # barrier = 2
            "physical_opportunity": 3,  # barrier = 2
            "planning": 3,  # barrier = 2
            "prioritization": 3,  # barrier = 2
            "social_opportunity": 3,  # barrier = 2
        }
        agent = ComBWeightedFixedAgent(comb_scores=scores, seed=42)
        counts: dict[str, int] = {}
        for _ in range(6000):
            action = agent.select_action(None)
            theme = action.rsplit("_", 1)[0]
            counts[theme] = counts.get(theme, 0) + 1
        # Each theme should appear ~1000 times; expect within 3 sigma (~160)
        for theme in scores:
            assert 600 <= counts.get(theme, 0) <= 1400, (
                f"{theme}: {counts.get(theme, 0)}"
            )

    def test_zero_barrier_themes_excluded(self):
        """Zero-barrier themes should never be selected."""
        scores = {
            "ability": 5,  # barrier = 0 → excluded
            "perceived_benefit": 1,  # barrier = 4
            "physical_opportunity": 5,  # barrier = 0 → excluded
            "planning": 1,  # barrier = 4
            "prioritization": 5,  # barrier = 0 → excluded
            "social_opportunity": 1,  # barrier = 4
        }
        agent = ComBWeightedFixedAgent(comb_scores=scores, seed=42)
        actions = [agent.select_action(None) for _ in range(3000)]
        themes = {a.rsplit("_", 1)[0] for a in actions}
        assert themes == {"perceived_benefit", "planning", "social_opportunity"}

    def test_distribution_matches_weights(self):
        """Theme distribution matches barrier weights over many samples."""
        scores = {
            "ability": 1,  # barrier = 4
            "perceived_benefit": 3,  # barrier = 2
            "physical_opportunity": 4,  # barrier = 1
            "planning": 2,  # barrier = 3
            "prioritization": 3,  # barrier = 2
            "social_opportunity": 3,  # barrier = 2
        }
        # Total barrier = 4+2+1+3+2+2 = 14
        # Expected: ability=4/14≈0.286, planning=3/14≈0.214, others=2/14≈0.143
        expected = {
            "ability": 4 / 14,
            "perceived_benefit": 2 / 14,
            "physical_opportunity": 1 / 14,
            "planning": 3 / 14,
            "prioritization": 2 / 14,
            "social_opportunity": 2 / 14,
        }
        agent = ComBWeightedFixedAgent(comb_scores=scores, seed=42)
        n_samples = 10000
        counts: dict[str, int] = {}
        for _ in range(n_samples):
            action = agent.select_action(None)
            theme = action.rsplit("_", 1)[0]
            counts[theme] = counts.get(theme, 0) + 1
        for theme, prob in expected.items():
            observed = counts.get(theme, 0) / n_samples
            # Allow ±3% tolerance
            assert abs(observed - prob) < 0.04, (
                f"{theme}: expected {prob:.3f}, got {observed:.3f}"
            )

    def test_all_zero_barriers_raises_value_error(self):
        """All themes scored 5 → all barriers zero → ValueError."""
        scores = dict.fromkeys(ComBWeightedFixedAgent.THEMES, 5)
        with pytest.raises(ValueError, match="all COM-B barriers are zero"):
            ComBWeightedFixedAgent(comb_scores=scores)

    @pytest.mark.parametrize(
        ("preference", "expected_morning_p", "tolerance"),
        [
            ("morning", 0.7, 0.06),
            ("afternoon", 0.3, 0.06),
            ("no_preference", 0.5, 0.06),
        ],
    )
    def test_timing_preference(self, preference, expected_morning_p, tolerance):
        """Timing distribution matches preference weights over many samples."""
        agent = ComBWeightedFixedAgent(
            comb_scores=_INLINE_SCORES,
            seed=42,
            time_preference=preference,
        )
        n_samples = 10000
        morning_count = 0
        for _ in range(n_samples):
            action = agent.select_action(None)
            if action.endswith("_morning"):
                morning_count += 1
        observed_p = morning_count / n_samples
        assert abs(observed_p - expected_morning_p) < tolerance, (
            f"time_preference={preference}: expected morning {expected_morning_p:.2f}, "
            f"got {observed_p:.3f}"
        )

    def test_default_time_preference(self):
        """No time_preference provided → 50/50 behavior."""
        agent = ComBWeightedFixedAgent(comb_scores=_INLINE_SCORES, seed=42)
        n_samples = 10000
        morning_count = 0
        for _ in range(n_samples):
            action = agent.select_action(None)
            if action.endswith("_morning"):
                morning_count += 1
        observed_p = morning_count / n_samples
        assert abs(observed_p - 0.5) < 0.06, (
            f"default time_preference: expected ~0.5, got {observed_p:.3f}"
        )

    def test_action_string_format(self):
        """Action string is '{theme}_{timing}'."""
        agent = ComBWeightedFixedAgent(comb_scores=_INLINE_SCORES, seed=42)
        for _ in range(100):
            action = agent.select_action(None)
            theme, timing = action.rsplit("_", 1)
            assert theme in ComBWeightedFixedAgent.THEMES
            assert timing in ComBWeightedFixedAgent.TIMINGS

    def test_all_12_actions_produced(self):
        """All 12 theme_timing combinations appear over many samples."""
        agent = ComBWeightedFixedAgent(comb_scores=_INLINE_SCORES, seed=42)
        observed: set[str] = set()
        for _ in range(5000):
            observed.add(agent.select_action(None))

        expected_themes = sorted(ComBWeightedFixedAgent.THEMES)
        expected_timings = sorted(ComBWeightedFixedAgent.TIMINGS)
        all_expected = {f"{t}_{tm}" for t in expected_themes for tm in expected_timings}
        assert observed == all_expected, (
            f"missing: {all_expected - observed}, extra: {observed - all_expected}"
        )

    def test_seed_reproducibility(self):
        """Same seed → identical 1000-action sequence."""
        agent_a = ComBWeightedFixedAgent(comb_scores=_INLINE_SCORES, seed=42)
        agent_b = ComBWeightedFixedAgent(comb_scores=_INLINE_SCORES, seed=42)
        seq_a = [agent_a.select_action(None) for _ in range(1000)]
        seq_b = [agent_b.select_action(None) for _ in range(1000)]
        assert seq_a == seq_b

    def test_different_seeds_different_sequences(self):
        """Different seeds → different sequences."""
        agent_a = ComBWeightedFixedAgent(comb_scores=_INLINE_SCORES, seed=42)
        agent_b = ComBWeightedFixedAgent(comb_scores=_INLINE_SCORES, seed=99)
        seq_a = [agent_a.select_action(None) for _ in range(100)]
        seq_b = [agent_b.select_action(None) for _ in range(100)]
        assert seq_a != seq_b

    def test_update_is_noop(self):
        """update() does not change agent behavior."""
        agent = ComBWeightedFixedAgent(comb_scores=_INLINE_SCORES, seed=42)
        first = agent.select_action(None)
        agent.update(None, "ability_morning", 1.0, None)
        second = agent.select_action(None)
        # Both should be valid action strings
        theme_a, timing_a = first.rsplit("_", 1)
        assert theme_a in ComBWeightedFixedAgent.THEMES
        assert timing_a in ComBWeightedFixedAgent.TIMINGS
        theme_b, timing_b = second.rsplit("_", 1)
        assert theme_b in ComBWeightedFixedAgent.THEMES
        assert timing_b in ComBWeightedFixedAgent.TIMINGS

    def test_on_day_end_is_noop(self):
        """on_day_end() does not change agent behavior."""
        agent = ComBWeightedFixedAgent(comb_scores=_INLINE_SCORES, seed=42)
        agent.on_day_end()
        action = agent.select_action(None)
        assert "_" in action
        theme, timing = action.rsplit("_", 1)
        assert theme in ComBWeightedFixedAgent.THEMES
        assert timing in ComBWeightedFixedAgent.TIMINGS

    def test_actions_none_defaults_to_all_12(self):
        """actions=None → generates all 12 combos."""
        agent = ComBWeightedFixedAgent(comb_scores=_INLINE_SCORES, seed=42)
        expected = sorted(
            f"{t}_{tm}"
            for t in sorted(ComBWeightedFixedAgent.THEMES)
            for tm in sorted(ComBWeightedFixedAgent.TIMINGS)
        )
        assert sorted(agent._actions) == expected

    def test_actions_with_idle(self):
        """Extra actions (like idle) are allowed but never selected."""
        all_12 = [
            f"{t}_{tm}"
            for t in sorted(ComBWeightedFixedAgent.THEMES)
            for tm in sorted(ComBWeightedFixedAgent.TIMINGS)
        ]
        agent = ComBWeightedFixedAgent(
            comb_scores=_INLINE_SCORES,
            seed=42,
            actions=[*all_12, "idle"],
        )
        # Over 1000 samples, only theme_timing actions are selected
        for _ in range(1000):
            action = agent.select_action(None)
            assert action != "idle"

    def test_actions_missing_one_raises(self):
        """Missing one theme_timing combo → ValueError."""
        actions = [
            f"{t}_{tm}"
            for t in sorted(ComBWeightedFixedAgent.THEMES)
            for tm in sorted(ComBWeightedFixedAgent.TIMINGS)
        ]
        actions.remove("ability_morning")
        with pytest.raises(ValueError, match="actions must contain all 12"):
            ComBWeightedFixedAgent(comb_scores=_INLINE_SCORES, actions=actions)

    def test_actions_missing_multiple_lists_all(self):
        """Missing multiple combos → error lists all missing."""
        actions = ["ability_morning", "ability_afternoon"]
        with pytest.raises(ValueError, match="actions must contain all 12"):
            ComBWeightedFixedAgent(comb_scores=_INLINE_SCORES, actions=actions)

    def test_actions_empty_list_raises(self):
        """Empty actions list → ValueError."""
        with pytest.raises(ValueError, match="actions must contain all 12"):
            ComBWeightedFixedAgent(comb_scores=_INLINE_SCORES, actions=[])


class TestBarrierComputation:
    def test_likert1_barrier4(self):
        """Likert=1 → barrier=4."""
        scores = {
            "ability": 1,
            "perceived_benefit": 5,
            "physical_opportunity": 5,
            "planning": 5,
            "prioritization": 5,
            "social_opportunity": 5,
        }
        agent = ComBWeightedFixedAgent(comb_scores=scores)
        assert agent._barriers["ability"] == 4

    def test_likert5_barrier0_excluded(self):
        """Likert=5 → barrier=0 → excluded from non-zero themes."""
        scores = {
            "ability": 5,
            "perceived_benefit": 5,
            "physical_opportunity": 5,
            "planning": 5,
            "prioritization": 5,
            "social_opportunity": 5,
        }
        with pytest.raises(ValueError, match="all COM-B barriers are zero"):
            ComBWeightedFixedAgent(comb_scores=scores)

    def test_likert3_barrier2(self):
        """Likert=3 → barrier=2."""
        scores = {
            "ability": 3,
            "perceived_benefit": 3,
            "physical_opportunity": 3,
            "planning": 3,
            "prioritization": 3,
            "social_opportunity": 3,
        }
        agent = ComBWeightedFixedAgent(comb_scores=scores)
        assert agent._barriers["ability"] == 2

    def test_all_ones_barriers_correct(self):
        """Likert=1 for all → barriers all 4."""
        scores = dict.fromkeys(ComBWeightedFixedAgent.THEMES, 1)
        agent = ComBWeightedFixedAgent(comb_scores=scores)
        for theme in sorted(ComBWeightedFixedAgent.THEMES):
            assert agent._barriers[theme] == 4, f"{theme} barrier != 4"


class TestScoreLoading:
    def test_file_based_valid_json(self, tmp_path):
        """Valid JSON file loads correctly."""
        data = {
            "test_persona": {
                "ability": 3,
                "perceived_benefit": 2,
                "physical_opportunity": 4,
                "planning": 2,
                "prioritization": 3,
                "social_opportunity": 3,
                "time_preference": "morning",
            }
        }
        fpath = tmp_path / "scores.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")
        agent = ComBWeightedFixedAgent(
            persona_comb_file=str(fpath),
            persona_name="test_persona",
        )
        action = agent.select_action(None)
        assert "_" in action

    def test_file_based_invalid_time_preference(self, tmp_path):
        """Invalid time_preference in JSON file → ValueError."""
        data = {
            "test_persona": {
                "ability": 3,
                "perceived_benefit": 2,
                "physical_opportunity": 4,
                "planning": 2,
                "prioritization": 3,
                "social_opportunity": 3,
                "time_preference": "night",
            }
        }
        fpath = tmp_path / "scores.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="invalid time_preference"):
            ComBWeightedFixedAgent(
                persona_comb_file=str(fpath),
                persona_name="test_persona",
            )

    def test_file_based_missing_file(self):
        """Missing file → FileNotFoundError."""
        with pytest.raises((FileNotFoundError, OSError)):
            ComBWeightedFixedAgent(
                persona_comb_file="/nonexistent/path/scores.json",
                persona_name="test",
            )

    def test_file_based_missing_persona(self, tmp_path):
        """Missing persona name → ValueError."""
        data = {
            "other_persona": {
                "ability": 3,
                "perceived_benefit": 2,
                "physical_opportunity": 4,
                "planning": 2,
                "prioritization": 3,
                "social_opportunity": 3,
            }
        }
        fpath = tmp_path / "scores.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match=r"persona.*not found"):
            ComBWeightedFixedAgent(
                persona_comb_file=str(fpath),
                persona_name="nonexistent_persona",
            )

    def test_file_based_missing_theme(self, tmp_path):
        """Missing theme → ValueError."""
        data = {"test": {"ability": 3, "perceived_benefit": 2}}
        fpath = tmp_path / "scores.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match=r"theme.*missing"):
            ComBWeightedFixedAgent(
                persona_comb_file=str(fpath),
                persona_name="test",
            )

    def test_file_based_out_of_range_score(self, tmp_path):
        """Out-of-range score → ValueError."""
        data = {
            "test": {
                "ability": 6,
                "perceived_benefit": 2,
                "physical_opportunity": 4,
                "planning": 2,
                "prioritization": 3,
                "social_opportunity": 3,
            }
        }
        fpath = tmp_path / "scores.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="must be an int in \\[1, 5\\]"):
            ComBWeightedFixedAgent(
                persona_comb_file=str(fpath),
                persona_name="test",
            )

    def test_inline_valid_dict(self):
        """Valid inline dict → agent created successfully."""
        agent = ComBWeightedFixedAgent(comb_scores=_INLINE_SCORES)
        assert agent.select_action(None) is not None

    def test_inline_empty_dict(self):
        """Empty inline dict → ValueError."""
        with pytest.raises(ValueError, match="comb_scores must be a non-empty dict"):
            ComBWeightedFixedAgent(comb_scores={})

    def test_inline_missing_theme(self):
        """Inline dict missing a theme → ValueError."""
        with pytest.raises(ValueError, match=r"theme.*missing from comb_scores"):
            ComBWeightedFixedAgent(comb_scores={"ability": 3, "perceived_benefit": 2})

    def test_both_provided_raises(self, tmp_path):
        """Both comb_scores and persona_comb_file → ValueError."""
        fpath = tmp_path / "scores.json"
        fpath.write_text(
            json.dumps(
                {
                    "x": {
                        "ability": 3,
                        "perceived_benefit": 2,
                        "physical_opportunity": 4,
                        "planning": 2,
                        "prioritization": 3,
                        "social_opportunity": 3,
                    }
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            ComBWeightedFixedAgent(
                comb_scores=_INLINE_SCORES,
                persona_comb_file=str(fpath),
                persona_name="x",
            )

    def test_neither_provided_raises(self):
        """Neither comb_scores nor persona_comb_file → ValueError."""
        with pytest.raises(ValueError, match="either comb_scores or persona_comb_file"):
            ComBWeightedFixedAgent()


class TestComBWeightedFixedConfig:
    def test_persona_file_and_name_valid(self, tmp_path):
        """persona_comb_file + persona_name → valid config."""
        data = {
            "p": {
                "ability": 3,
                "perceived_benefit": 2,
                "physical_opportunity": 4,
                "planning": 2,
                "prioritization": 3,
                "social_opportunity": 3,
            }
        }
        fpath = tmp_path / "s.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")
        cfg = AgentConfig.model_validate(
            {
                "type": "comb_weighted_fixed",
                "persona_comb_file": str(fpath),
                "persona_name": "p",
            }
        )
        assert cfg.type == "comb_weighted_fixed"

    def test_inline_scores_only_valid(self):
        """inline_comb_scores only → valid config."""
        cfg = AgentConfig.model_validate(
            {
                "type": "comb_weighted_fixed",
                "inline_comb_scores": _INLINE_SCORES,
            }
        )
        assert cfg.type == "comb_weighted_fixed"

    def test_both_raises_value_error(self):
        """Both persona_comb_file/persona_name and inline_comb_scores → ValueError."""
        with pytest.raises(ValueError, match="not both"):
            AgentConfig.model_validate(
                {
                    "type": "comb_weighted_fixed",
                    "persona_comb_file": "test.json",
                    "persona_name": "p",
                    "inline_comb_scores": _INLINE_SCORES,
                }
            )

    def test_neither_raises_value_error(self):
        """Neither → ValueError."""
        with pytest.raises(ValueError, match="requires either"):
            AgentConfig.model_validate({"type": "comb_weighted_fixed"})

    @pytest.mark.parametrize("pref", ["morning", "afternoon", "no_preference"])
    def test_valid_time_preferences(self, pref):
        """Valid time_preference values."""
        cfg = AgentConfig.model_validate(
            {
                "type": "comb_weighted_fixed",
                "inline_comb_scores": _INLINE_SCORES,
                "time_preference": pref,
            }
        )
        assert cfg.time_preference == pref

    def test_invalid_time_preference_raises(self):
        """Invalid time_preference → ValueError."""
        with pytest.raises(ValueError, match="time_preference"):
            AgentConfig.model_validate(
                {
                    "type": "comb_weighted_fixed",
                    "inline_comb_scores": _INLINE_SCORES,
                    "time_preference": "night",
                }
            )

    def test_empty_time_preference_raises(self):
        """Empty time_preference string → ValueError."""
        with pytest.raises(ValueError, match="time_preference"):
            AgentConfig.model_validate(
                {
                    "type": "comb_weighted_fixed",
                    "inline_comb_scores": _INLINE_SCORES,
                    "time_preference": "",
                }
            )

    @pytest.mark.parametrize(
        "field",
        [
            {"alpha_prior": 1.0},
            {"beta_prior": 1.0},
            {"epsilon": 0.1},
            {"epsilon_start": 0.2},
            {"c": 2.0},
            {"action": "idle"},
            {"contextual": True},
        ],
    )
    def test_rejects_learning_params(self, field):
        """comb_weighted_fixed rejects learning hyperparameters."""
        with pytest.raises(ValueError, match="not applicable"):
            AgentConfig.model_validate(
                {
                    "type": "comb_weighted_fixed",
                    "inline_comb_scores": _INLINE_SCORES,
                    **field,
                }
            )

    def test_empty_persona_comb_file_raises(self):
        """Empty persona_comb_file string → rejected."""
        with pytest.raises(ValueError, match="non-empty string"):
            AgentConfig.model_validate(
                {
                    "type": "comb_weighted_fixed",
                    "persona_comb_file": "",
                    "persona_name": "p",
                }
            )

    def test_empty_persona_name_raises(self):
        """Empty persona_name string → rejected."""
        with pytest.raises(ValueError, match="non-empty string"):
            AgentConfig.model_validate(
                {
                    "type": "comb_weighted_fixed",
                    "persona_comb_file": "test.json",
                    "persona_name": "",
                }
            )

    def test_persona_name_without_file_raises(self):
        """persona_name alone → rejected (incomplete pair)."""
        with pytest.raises(ValueError, match="must be provided together"):
            AgentConfig.model_validate(
                {
                    "type": "comb_weighted_fixed",
                    "persona_name": "p",
                }
            )

    def test_persona_file_without_name_raises(self):
        """persona_comb_file without persona_name → rejected (incomplete pair)."""
        with pytest.raises(ValueError, match="must be provided together"):
            AgentConfig.model_validate(
                {
                    "type": "comb_weighted_fixed",
                    "persona_comb_file": "test.json",
                }
            )
