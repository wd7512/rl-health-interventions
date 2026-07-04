from __future__ import annotations

import pytest
from pydantic import ValidationError
from rl_health_interventions.config.schemas import MDPConfig


def _valid_raw():
    return {
        "episode_days": 90,
        "steps_per_day": 5,
        "seed": 42,
        "state": {
            "variables": {
                "activity_level": {"dims": 2, "names": ["sedentary", "active"]}
            }
        },
        "initial_state": {"activity_level": "sedentary"},
        "actions": ["nudge", "idle"],
        "reward": {
            "variables": {
                "value": {
                    "source": "state.activity_level",
                    "mapping": {"sedentary": 0.0, "active": 1.0},
                }
            },
            "formula": "value",
        },
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


class TestInitialStateValidator:
    def test_initial_state_key_not_in_variables_rejected(self):
        raw = _valid_raw()
        raw["initial_state"] = {"nonexistent": "value"}
        with pytest.raises(ValidationError, match="not found"):
            MDPConfig.model_validate(raw)

    def test_initial_state_value_not_in_variable_names_rejected(self):
        raw = _valid_raw()
        raw["initial_state"] = {"activity_level": "nonexistent"}
        with pytest.raises(ValidationError, match="not in"):
            MDPConfig.model_validate(raw)


class TestRewardValidator:
    def test_unknown_state_variable_in_reward_source_rejected(self):
        raw = _valid_raw()
        raw["reward"]["variables"]["bad"] = {
            "source": "state.nonexistent",
            "mapping": {"x": 0.0},
        }
        with pytest.raises(ValidationError, match="unknown state variable"):
            MDPConfig.model_validate(raw)

    def test_constant_variable_name_conflict_rejected(self):
        raw = _valid_raw()
        raw["reward"]["constants"] = {"value": 0.5}
        with pytest.raises(ValidationError, match="conflict"):
            MDPConfig.model_validate(raw)


class TestCrossReferenceValidators:
    def test_rule_based_requires_transition_probabilities(self):
        raw = _valid_raw()
        del raw["transition_model"]["transition_probabilities"]
        with pytest.raises(ValidationError, match="must be provided for rule_based"):
            MDPConfig.model_validate(raw)

    def test_transition_probs_must_cover_all_variable_values(self):
        raw = _valid_raw()
        del raw["transition_model"]["transition_probabilities"]["active"]
        with pytest.raises(
            ValidationError, match="missing from transition_probabilities"
        ):
            MDPConfig.model_validate(raw)

    def test_transition_target_not_in_variable_names_rejected(self):
        raw = _valid_raw()
        raw["transition_model"]["transition_probabilities"]["sedentary"]["nudge"] = {
            "nonexistent": 0.7,
            "active": 0.3,
        }
        with pytest.raises(ValidationError, match="not in"):
            MDPConfig.model_validate(raw)

    def test_incomplete_target_distribution_rejected(self):
        raw = _valid_raw()
        raw["transition_model"]["transition_probabilities"]["sedentary"]["nudge"] = {
            "active": 1.0,
        }
        with pytest.raises(ValidationError, match="cover all"):
            MDPConfig.model_validate(raw)

    def test_missing_action_entry_rejected(self):
        raw = _valid_raw()
        del raw["transition_model"]["transition_probabilities"]["sedentary"]["idle"]
        with pytest.raises(ValidationError, match="Missing transition"):
            MDPConfig.model_validate(raw)


class TestRewardMultiplier:
    def test_multiplier_len_mismatch_rejected(self):
        raw = _valid_raw()
        raw["reward_multiplier_by_step"] = [1, 1]
        with pytest.raises(ValidationError, match="length"):
            MDPConfig.model_validate(raw)


class TestAgentValidation:
    def test_known_agent_types_accepted(self):
        raw = _valid_raw()
        raw["agents"] = [
            {"type": "thompson_sampling", "alpha_prior": 1, "beta_prior": 1},
            {"type": "random"},
            {"type": "epsilon_greedy", "epsilon": 0.1},
            {"type": "ucb", "c": 2.0},
            {"type": "fixed"},
        ]
        config = MDPConfig.model_validate(raw)
        assert len(config.agents) == 5

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

    def test_contextual_accepted_for_bandits(self):
        raw = _valid_raw()
        raw["agents"] = [
            {
                "type": "thompson_sampling",
                "alpha_prior": 1,
                "beta_prior": 1,
                "contextual": True,
                "context_feature": "activity_level",
            },
        ]
        config = MDPConfig.model_validate(raw)
        assert len(config.agents) == 1
        assert config.agents[0].contextual
        assert config.agents[0].context_feature == "activity_level"

    def test_contextual_rejected_for_random(self):
        raw = _valid_raw()
        raw["agents"] = [
            {"type": "random", "contextual": True, "context_feature": "activity_level"}
        ]
        with pytest.raises(ValidationError, match="contextual=True is only supported"):
            MDPConfig.model_validate(raw)

    def test_non_contextual_rejects_context_feature(self):
        raw = _valid_raw()
        raw["agents"] = [
            {
                "type": "epsilon_greedy",
                "epsilon": 0.1,
                "context_feature": "activity_level",
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


class TestNonRuleBasedTransition:
    def test_non_rule_based_without_probabilities_accepted(self):
        raw = _valid_raw()
        raw["transition_model"] = {"type": "learned"}
        config = MDPConfig.model_validate(raw)
        assert config.transition_model.type == "learned"


class TestActionsCoercion:
    def test_list_actions_coerced_to_dict(self):
        raw = _valid_raw()
        config = MDPConfig.model_validate(raw)
        assert isinstance(config.actions, dict)
        assert "nudge" in config.actions
        assert "idle" in config.actions

    def test_dict_actions_accepted_directly(self):
        raw = _valid_raw()
        raw["actions"] = {"nudge": {}, "idle": {}}
        config = MDPConfig.model_validate(raw)
        assert config.action_names == ["nudge", "idle"]


class TestFixedAgent:
    def test_fixed_agent_accepted(self):
        raw = _valid_raw()
        raw["agents"] = [{"type": "fixed", "action": "idle"}]
        config = MDPConfig.model_validate(raw)
        assert config.agents[0].type == "fixed"
        assert config.agents[0].action == "idle"

    def test_fixed_agent_skips_contextual_validation(self):
        raw = _valid_raw()
        raw["agents"] = [{"type": "fixed", "contextual": True}]
        config = MDPConfig.model_validate(raw)
        assert config.agents[0].type == "fixed"


class TestContextFeatureList:
    def test_list_context_feature_accepted(self):
        raw = _valid_raw()
        raw["agents"] = [
            {
                "type": "thompson_sampling",
                "alpha_prior": 1,
                "beta_prior": 1,
                "contextual": True,
                "context_feature": ["step_bin", "burden"],
            }
        ]
        config = MDPConfig.model_validate(raw)
        assert config.agents[0].context_feature == ["step_bin", "burden"]
