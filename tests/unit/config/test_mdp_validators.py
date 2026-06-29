from __future__ import annotations

import pytest
from pydantic import ValidationError

from rl_health_interventions.config.schemas import (
    MDPConfig,
    AgentConfig,
    ActionConfig,
    FactorConfig,
    RewardConfig,
    StateConfig,
    TransitionModelConfig,
)


def _valid_raw():
    """New format valid raw config for testing."""
    return {
        "episode_days": 90,
        "steps_per_day": 5,
        "seed": 42,
        "initial_state": {"activity_level": "sedentary"},
        "state": {
            "factors": {
                "activity_level": {"dims": 2, "names": ["sedentary", "active"]},
            },
        },
        "actions": ["nudge", "idle"],
        "reward": {
            "factor": "activity_level",
            "values": {"sedentary": 0.0, "active": 1.0},
        },
        "transition_model": {"type": "rule_based", "table_dir": "tables"},
        "agents": [{"type": "thompson_sampling", "alpha_prior": 1, "beta_prior": 1}],
    }


class TestActionsValidation:
    def test_actions_list_auto_converted_to_dict(self):
        raw = _valid_raw()
        config = MDPConfig.model_validate(raw)
        assert isinstance(config.actions, dict)
        assert "nudge" in config.actions
        assert "idle" in config.actions

    def test_actions_empty_list_converted(self):
        raw = _valid_raw()
        raw["actions"] = []
        config = MDPConfig.model_validate(raw)
        assert config.actions == {}


class TestStateConfigValidation:
    def test_state_must_have_factors_or_schema(self):
        raw = _valid_raw()
        raw["state"] = {}
        with pytest.raises(ValidationError):
            MDPConfig.model_validate(raw)

    def test_state_must_not_have_both_factors_and_schema(self):
        raw = _valid_raw()
        raw["state"] = {
            "factors": {"x": {"dims": 1, "names": ["a"]}},
            "schema": "test",
        }
        with pytest.raises(ValidationError):
            MDPConfig.model_validate(raw)


class TestCrossReferenceValidators:
    def test_initial_state_keys_match_factor_names(self):
        raw = _valid_raw()
        raw["initial_state"] = {"wrong_key": "sedentary"}
        with pytest.raises(ValidationError, match="initial_state"):
            MDPConfig.model_validate(raw)

    def test_initial_state_missing_factor_key(self):
        raw = _valid_raw()
        raw["initial_state"] = {}
        with pytest.raises(ValidationError, match="initial_state"):
            MDPConfig.model_validate(raw)

    def test_initial_state_extra_factor_key(self):
        raw = _valid_raw()
        raw["initial_state"] = {"activity_level": "sedentary", "extra": "value"}
        with pytest.raises(ValidationError, match="initial_state"):
            MDPConfig.model_validate(raw)

    def test_reward_factor_not_in_state_factors_rejected(self):
        raw = _valid_raw()
        raw["reward"]["factor"] = "nonexistent"
        with pytest.raises(ValidationError, match="factor"):
            MDPConfig.model_validate(raw)

    def test_reward_values_keys_do_not_match_factor_names_rejected(self):
        raw = _valid_raw()
        raw["reward"]["values"] = {"wrong": 0.0}
        with pytest.raises(ValidationError, match="values"):
            MDPConfig.model_validate(raw)

    def test_reward_values_missing_one_key_rejected(self):
        raw = _valid_raw()
        raw["reward"]["values"] = {"sedentary": 0.0}
        with pytest.raises(ValidationError, match="values"):
            MDPConfig.model_validate(raw)

    def test_initial_state_value_invalid_for_factor_rejected(self):
        raw = _valid_raw()
        raw["initial_state"]["activity_level"] = "bogus"
        with pytest.raises(ValidationError, match="initial_state|bogus"):
            MDPConfig.model_validate(raw)


class TestPerStepMultiplier:
    def test_multiplier_not_required(self):
        raw = _valid_raw()
        config = MDPConfig.model_validate(raw)
        assert config.reward.per_step_multiplier is None

    def test_multiplier_len_mismatch_rejected(self):
        raw = _valid_raw()
        raw["reward"]["per_step_multiplier"] = [1, 2, 3]
        with pytest.raises(ValidationError, match="per_step_multiplier|length"):
            MDPConfig.model_validate(raw)

    def test_multiplier_matches_steps_per_day(self):
        raw = _valid_raw()
        raw["reward"]["per_step_multiplier"] = [1.0, 1.0, 1.0, 1.0, 0.0]
        config = MDPConfig.model_validate(raw)
        assert config.reward.per_step_multiplier == [1.0, 1.0, 1.0, 1.0, 0.0]


class TestSchemaRefMode:
    def test_schema_ref_skips_cross_references(self):
        raw = _valid_raw()
        raw["state"] = {"schema": "heartsteps"}
        raw["actions"] = {"schema": "heartsteps"}
        del raw["transition_model"]["table_dir"]
        raw["transition_model"]["type"] = "learned"
        config = MDPConfig.model_validate(raw)
        assert config.state.schema == "heartsteps"

    def test_schema_ref_with_actions_list(self):
        raw = _valid_raw()
        raw["state"] = {"schema": "heartsteps"}
        raw["actions"] = ["nudge", "idle"]
        del raw["transition_model"]["table_dir"]
        raw["transition_model"]["type"] = "learned"
        config = MDPConfig.model_validate(raw)
        assert config.state.schema == "heartsteps"


class TestFactorConfig:
    def test_dims_positive(self):
        raw = _valid_raw()
        raw["state"]["factors"]["bad"] = {"dims": 0, "names": []}
        with pytest.raises(ValidationError):
            MDPConfig.model_validate(raw)

    def test_names_length_must_match_dims(self):
        raw = _valid_raw()
        raw["state"]["factors"]["bad"] = {"dims": 3, "names": ["a", "b"]}
        with pytest.raises(ValidationError):
            MDPConfig.model_validate(raw)

    def test_boundaries_optional(self):
        raw = _valid_raw()
        raw["state"]["factors"]["activity_level"]["boundaries"] = [500.0]
        config = MDPConfig.model_validate(raw)
        assert config.factor_configs["activity_level"].boundaries == [500.0]

    def test_boundaries_can_be_none(self):
        raw = _valid_raw()
        config = MDPConfig.model_validate(raw)
        assert config.factor_configs["activity_level"].boundaries is None


class TestTransitionModelConfig:
    def test_rule_based_with_table_dir_accepted(self):
        raw = _valid_raw()
        config = MDPConfig.model_validate(raw)
        assert config.transition_model.type == "rule_based"
        assert config.transition_model.table_dir == "tables"

    def test_bootstrap_type_accepted(self):
        raw = _valid_raw()
        raw["transition_model"] = {"type": "bootstrap", "table_dir": "../tables"}
        config = MDPConfig.model_validate(raw)
        assert config.transition_model.type == "bootstrap"

    def test_unknown_type_rejected(self):
        raw = _valid_raw()
        raw["transition_model"] = {"type": "unknown_type"}
        with pytest.raises(ValidationError, match="Unknown transition type"):
            MDPConfig.model_validate(raw)

    def test_learned_type_without_table_dir_accepted(self):
        raw = _valid_raw()
        raw["transition_model"] = {"type": "learned"}
        config = MDPConfig.model_validate(raw)
        assert config.transition_model.type == "learned"


class TestActionConfig:
    def test_action_penalty_default(self):
        raw = _valid_raw()
        raw["actions"] = ["nudge"]
        config = MDPConfig.model_validate(raw)
        assert config.actions["nudge"].action_penalty == 0.0

    def test_action_penalty_custom(self):
        raw = _valid_raw()
        raw["actions"] = {"nudge": {"action_penalty": 1.5}}
        config = MDPConfig.model_validate(raw)
        assert config.actions["nudge"].action_penalty == 1.5


class TestModelConstruction:
    def test_factor_config_direct(self):
        fc = FactorConfig(dims=3, names=["a", "b", "c"])
        assert fc.dims == 3
        assert fc.names == ["a", "b", "c"]
        assert fc.boundaries is None

    def test_action_config_direct(self):
        ac = ActionConfig(action_penalty=0.5)
        assert ac.action_penalty == 0.5

    def test_transition_model_config_direct(self):
        tmc = TransitionModelConfig(type="bootstrap", table_dir="../tables")
        assert tmc.type == "bootstrap"
        assert tmc.table_dir == "../tables"

    def test_transition_model_table_dir_required_for_table_backed(self):
        with pytest.raises(ValidationError, match="table_dir"):
            TransitionModelConfig(type="rule_based")

    def test_reward_config_direct(self):
        rc = RewardConfig(
            factor="activity_level", values={"sedentary": 0.0, "active": 1.0}
        )
        assert rc.factor == "activity_level"
        assert rc.action_penalty_multiplier == 0.0
        assert rc.per_step_multiplier is None

    def test_state_config_with_factors(self):
        sc = StateConfig(factors={"x": FactorConfig(dims=1, names=["a"])})
        assert sc.factors is not None
        assert sc.schema is None

    def test_state_config_with_schema(self):
        sc = StateConfig(schema="heartsteps")
        assert sc.factors is None
        assert sc.schema == "heartsteps"

    def test_agent_config_direct(self):
        ac = AgentConfig(
            type="thompson_sampling",
            alpha_prior=2.0,
            beta_prior=5.0,
            contextual=True,
            context_feature="activity_level",
        )
        assert ac.type == "thompson_sampling"
        assert ac.alpha_prior == 2.0
        assert ac.contextual is True


class TestOldFormatRejected:
    """Old states dict format should not be accepted."""

    def test_old_states_format_rejected(self):
        raw = {
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
        }
        with pytest.raises(ValidationError):
            MDPConfig.model_validate(raw)

    def test_old_string_initial_state_rejected(self):
        """initial_state as string (old format) should be rejected."""
        raw = {
            "episode_days": 1,
            "steps_per_day": 1,
            "seed": 42,
            "initial_state": "sedentary",
            "state": {
                "factors": {
                    "activity_level": {"dims": 2, "names": ["sedentary", "active"]},
                },
            },
            "actions": ["nudge"],
            "reward": {
                "factor": "activity_level",
                "values": {"sedentary": 0.0, "active": 1.0},
            },
            "transition_model": {"type": "rule_based", "table_dir": "tables"},
        }
        with pytest.raises(ValidationError):
            MDPConfig.model_validate(raw)


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

    def test_random_rejects_all_hyperparameters(self):
        raw = _valid_raw()
        raw["agents"] = [{"type": "random", "epsilon": 0.1}]
        with pytest.raises(ValidationError, match="does not accept any"):
            MDPConfig.model_validate(raw)
