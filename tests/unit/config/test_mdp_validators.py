from __future__ import annotations

import pytest
from pydantic import ValidationError
from rl_health_interventions.config.schemas import MDPConfig


def _valid_raw():
    return {
        "episode_days": 90,
        "steps_per_day": 5,
        "seed": 42,
        "initial_state": "sedentary",
        "states": {"sedentary": {"reward": 0.0}, "active": {"reward": 1.0}},
        "actions": ["nudge", "idle"],
        "transition_model": {
            "type": "rule_based",
            "transition_probabilities": {
                "sedentary": {
                    "nudge": {"active": 0.3, "sedentary": 0.7},
                    "idle": {"active": 0.1, "sedentary": 0.9},
                },
                "active": {
                    "nudge": {"active": 0.5, "sedentary": 0.5},
                    "idle": {"active": 0.6, "sedentary": 0.4},
                },
            },
        },
        "agents": [{"type": "thompson_sampling", "alpha_prior": 1, "beta_prior": 1}],
    }


class TestTransitionProbabilities:
    def test_negative_probability_rejected(self):
        raw = _valid_raw()
        raw["transition_model"]["transition_probabilities"]["sedentary"]["nudge"][
            "active"
        ] = -0.5
        with pytest.raises(ValidationError, match="negative"):
            MDPConfig.model_validate(raw)

    def test_probabilities_not_summing_to_one_rejected(self):
        raw = _valid_raw()
        raw["transition_model"]["transition_probabilities"]["sedentary"]["nudge"][
            "active"
        ] = 0.5
        with pytest.raises(ValidationError, match="sum"):
            MDPConfig.model_validate(raw)


class TestCrossReferenceValidators:
    def test_empty_states_rejected(self):
        raw = _valid_raw()
        raw["states"] = {}
        with pytest.raises(ValidationError, match="empty"):
            MDPConfig.model_validate(raw)

    def test_state_missing_reward_rejected(self):
        raw = _valid_raw()
        raw["states"]["sedentary"] = {}
        with pytest.raises(ValidationError, match="reward"):
            MDPConfig.model_validate(raw)

    def test_empty_actions_rejected(self):
        raw = _valid_raw()
        raw["actions"] = []
        with pytest.raises(ValidationError, match="empty"):
            MDPConfig.model_validate(raw)

    def test_duplicate_actions_rejected(self):
        raw = _valid_raw()
        raw["actions"] = ["nudge", "nudge"]
        with pytest.raises(ValidationError, match="duplicate"):
            MDPConfig.model_validate(raw)

    def test_transition_state_not_in_states_rejected(self):
        raw = _valid_raw()
        raw["transition_model"]["transition_probabilities"]["unknown"] = {
            "nudge": {"active": 0.5, "sedentary": 0.5}
        }
        with pytest.raises(ValidationError, match="unknown"):
            MDPConfig.model_validate(raw)

    def test_missing_action_entry_rejected(self):
        raw = _valid_raw()
        del raw["transition_model"]["transition_probabilities"]["sedentary"]["idle"]
        with pytest.raises(ValidationError, match="idle"):
            MDPConfig.model_validate(raw)

    def test_transition_target_not_in_states_rejected(self):
        raw = _valid_raw()
        raw["transition_model"]["transition_probabilities"]["sedentary"]["nudge"] = {
            "unknown": 0.7,
            "active": 0.3,
        }
        with pytest.raises(ValidationError, match="unknown"):
            MDPConfig.model_validate(raw)

    def test_incomplete_target_distribution_rejected(self):
        raw = _valid_raw()
        raw["transition_model"]["transition_probabilities"]["sedentary"]["nudge"] = {
            "active": 1.0,
        }
        with pytest.raises(ValidationError, match="cover"):
            MDPConfig.model_validate(raw)


class TestRewardMultiplier:
    def test_multiplier_defaults_to_uniform(self):
        raw = _valid_raw()
        config = MDPConfig.model_validate(raw)
        assert config.reward_multiplier_by_step is None
        assert config.per_step_reward is not None
        assert len(config.per_step_reward) == 5
        for step_reward in config.per_step_reward:
            assert step_reward["active"] == 1.0
            assert step_reward["sedentary"] == 0.0

    def test_multiplier_len_mismatch_rejected(self):
        raw = _valid_raw()
        raw["reward_multiplier_by_step"] = [1, 1]
        with pytest.raises(ValidationError, match="length"):
            MDPConfig.model_validate(raw)

    def test_multiplier_zero_masks_step(self):
        raw = _valid_raw()
        raw["reward_multiplier_by_step"] = [1, 1, 1, 1, 0]
        config = MDPConfig.model_validate(raw)
        assert config.per_step_reward is not None
        assert config.per_step_reward[4]["active"] == 0.0


class TestSchemaRefMode:
    def test_schema_ref_states_skips_inline_checks(self):
        raw = _valid_raw()
        raw["states"] = {"schema": "heartsteps"}
        raw["actions"] = {"schema": "heartsteps"}
        del raw["transition_model"]["transition_probabilities"]
        config = MDPConfig.model_validate(raw)
        assert "schema" in config.states


class TestAgentValidation:
    def test_known_agent_types_accepted(self):
        raw = _valid_raw()
        raw["agents"] = [
            {"type": "thompson_sampling", "alpha_prior": 1, "beta_prior": 1},
            {"type": "random"},
            {"type": "epsilon_greedy", "epsilon": 0.1},
            {"type": "ucb", "c": 2.0},
        ]
        config = MDPConfig.model_validate(raw)
        assert len(config.agents) == 4

    def test_thompson_sampling_requires_positive_priors(self):
        raw = _valid_raw()
        raw["agents"] = [
            {"type": "thompson_sampling", "alpha_prior": 0, "beta_prior": 1}
        ]
        with pytest.raises(ValidationError, match="alpha_prior"):
            MDPConfig.model_validate(raw)

    def test_epsilon_greedy_requires_epsilon_in_range(self):
        raw = _valid_raw()
        raw["agents"] = [{"type": "epsilon_greedy", "epsilon": 1.5}]
        with pytest.raises(ValidationError, match="epsilon"):
            MDPConfig.model_validate(raw)

    def test_ucb_requires_positive_c(self):
        raw = _valid_raw()
        raw["agents"] = [{"type": "ucb", "c": 0}]
        with pytest.raises(ValidationError, match="c"):
            MDPConfig.model_validate(raw)

    def test_unknown_agent_type_rejected(self):
        raw = _valid_raw()
        raw["agents"] = [{"type": "unknown_agent"}]
        with pytest.raises(ValidationError, match="unknown"):
            MDPConfig.model_validate(raw)
