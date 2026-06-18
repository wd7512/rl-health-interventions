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

    def test_state_non_numeric_reward_rejected(self):
        raw = _valid_raw()
        raw["states"]["sedentary"] = {"reward": "not_a_number"}
        with pytest.raises(ValidationError, match="numeric"):
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

    def test_thompson_sampling_rejects_epsilon(self):
        raw = _valid_raw()
        raw["agents"] = [
            {
                "type": "thompson_sampling",
                "alpha_prior": 1,
                "beta_prior": 1,
                "epsilon": 0.1,
            }
        ]
        with pytest.raises(ValidationError, match="does not accept epsilon"):
            MDPConfig.model_validate(raw)

    def test_epsilon_greedy_rejects_alpha_prior(self):
        raw = _valid_raw()
        raw["agents"] = [
            {
                "type": "epsilon_greedy",
                "epsilon": 0.1,
                "alpha_prior": 1,
                "beta_prior": 1,
            }
        ]
        with pytest.raises(ValidationError, match="does not accept alpha_prior"):
            MDPConfig.model_validate(raw)

    def test_ucb_rejects_alpha_prior(self):
        raw = _valid_raw()
        raw["agents"] = [{"type": "ucb", "c": 2.0, "alpha_prior": 1, "beta_prior": 1}]
        with pytest.raises(ValidationError, match="does not accept alpha_prior"):
            MDPConfig.model_validate(raw)

    def test_contextual_accepted_for_bandits(self):
        raw = _valid_raw()
        raw["agents"] = [
            {
                "type": "thompson_sampling",
                "alpha_prior": 1,
                "beta_prior": 1,
                "contextual": True,
                "context_feature": "activity",
            },
            {
                "type": "epsilon_greedy",
                "epsilon": 0.1,
                "contextual": True,
                "context_feature": "activity",
            },
            {
                "type": "ucb",
                "c": 2.0,
                "contextual": True,
                "context_feature": "activity",
            },
        ]
        config = MDPConfig.model_validate(raw)
        assert len(config.agents) == 3
        for agent_cfg in config.agents:
            assert agent_cfg.contextual
            assert agent_cfg.context_feature == "activity"

    def test_contextual_rejected_for_random(self):
        raw = _valid_raw()
        raw["agents"] = [
            {"type": "random", "contextual": True, "context_feature": "activity"}
        ]
        with pytest.raises(ValidationError, match="contextual=True is only supported"):
            MDPConfig.model_validate(raw)

    def test_contextual_requires_context_feature(self):
        raw = _valid_raw()
        raw["agents"] = [{"type": "epsilon_greedy", "epsilon": 0.1, "contextual": True}]
        with pytest.raises(
            ValidationError, match="context_feature must be a non-empty string"
        ):
            MDPConfig.model_validate(raw)

    def test_contextual_with_empty_context_feature_rejected(self):
        raw = _valid_raw()
        raw["agents"] = [
            {
                "type": "epsilon_greedy",
                "epsilon": 0.1,
                "contextual": True,
                "context_feature": "",
            }
        ]
        with pytest.raises(
            ValidationError, match="context_feature must be a non-empty string"
        ):
            MDPConfig.model_validate(raw)

    def test_contextual_with_whitespace_context_feature_rejected(self):
        raw = _valid_raw()
        raw["agents"] = [
            {
                "type": "epsilon_greedy",
                "epsilon": 0.1,
                "contextual": True,
                "context_feature": "   ",
            }
        ]
        with pytest.raises(
            ValidationError, match="context_feature must be a non-empty string"
        ):
            MDPConfig.model_validate(raw)

    def test_non_contextual_rejects_context_feature(self):
        raw = _valid_raw()
        raw["agents"] = [
            {
                "type": "epsilon_greedy",
                "epsilon": 0.1,
                "context_feature": "activity",
            }
        ]
        with pytest.raises(
            ValidationError, match="context_feature must not be provided"
        ):
            MDPConfig.model_validate(raw)

    def test_random_rejects_all_hyperparameters(self):
        raw = _valid_raw()
        raw["agents"] = [{"type": "random", "epsilon": 0.1}]
        with pytest.raises(
            ValidationError, match="does not accept any hyperparameters"
        ):
            MDPConfig.model_validate(raw)

    def test_rule_based_requires_transition_probabilities(self):
        raw = _valid_raw()
        del raw["transition_model"]["transition_probabilities"]
        with pytest.raises(ValidationError, match="must be provided for rule_based"):
            MDPConfig.model_validate(raw)

    def test_transition_probs_must_cover_all_states(self):
        raw = _valid_raw()
        del raw["transition_model"]["transition_probabilities"]["active"]
        with pytest.raises(
            ValidationError, match="missing from transition_probabilities"
        ):
            MDPConfig.model_validate(raw)

    def test_states_must_be_dict(self):
        raw = _valid_raw()
        raw["states"] = ["sedentary", "active"]
        with pytest.raises(ValidationError, match="must be a dictionary"):
            MDPConfig.model_validate(raw)

    def test_actions_must_be_list(self):
        raw = _valid_raw()
        raw["actions"] = {"nudge": 1, "idle": 0}
        with pytest.raises(ValidationError, match="must be a list"):
            MDPConfig.model_validate(raw)

    def test_initial_state_not_in_states_rejected(self):
        raw = _valid_raw()
        raw["initial_state"] = "nonexistent"
        with pytest.raises(ValidationError, match="initial_state.*not in states"):
            MDPConfig.model_validate(raw)


class TestSchemaRefActions:
    def test_schema_ref_actions_skips_inline_checks(self):
        raw = _valid_raw()
        raw["states"] = {"sedentary": {"reward": 0.0}, "active": {"reward": 1.0}}
        raw["actions"] = {"schema": "heartsteps"}
        del raw["transition_model"]["transition_probabilities"]
        config = MDPConfig.model_validate(raw)
        assert isinstance(config.actions, dict)
        assert "schema" in config.actions


class TestNonRuleBasedTransition:
    def test_non_rule_based_without_probabilities_accepted(self):
        raw = _valid_raw()
        raw["transition_model"] = {"type": "learned"}
        config = MDPConfig.model_validate(raw)
        assert config.transition_model.type == "learned"


class TestDecayingEpsilonGreedy:
    def test_valid_config_accepted(self):
        raw = _valid_raw()
        raw["agents"] = [
            {
                "type": "decaying_epsilon_greedy",
                "epsilon_start": 0.5,
                "epsilon_min": 0.01,
                "decay_steps": 200,
            }
        ]
        config = MDPConfig.model_validate(raw)
        assert config.agents[0].type == "decaying_epsilon_greedy"

    def test_epsilon_min_exceeding_start_rejected(self):
        raw = _valid_raw()
        raw["agents"] = [
            {
                "type": "decaying_epsilon_greedy",
                "epsilon_start": 0.1,
                "epsilon_min": 0.5,
            }
        ]
        with pytest.raises(
            ValidationError, match="epsilon_min must not exceed epsilon_start"
        ):
            MDPConfig.model_validate(raw)

    def test_epsilon_start_out_of_range_rejected(self):
        raw = _valid_raw()
        raw["agents"] = [{"type": "decaying_epsilon_greedy", "epsilon_start": 1.5}]
        with pytest.raises(ValidationError, match="epsilon_start must be in"):
            MDPConfig.model_validate(raw)
