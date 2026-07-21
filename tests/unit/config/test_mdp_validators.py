from __future__ import annotations

import pytest
from pydantic import ValidationError

from rl_health_interventions.config.schemas import MDPConfig


def _valid_raw():
    return {
        "episode_days": 90,
        "steps_per_day": 5,
        "seed": 42,
        "state": {"variables": {"activity_level": {"names": ["sedentary", "active"]}}},
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


# Helpers for compact lambda mutations in parametrize
def _set(raw, *keys_val):
    keys, val = keys_val[:-1], keys_val[-1]
    for k in keys[:-1]:
        raw = raw[k]
    raw[keys[-1]] = val


def _del(raw, *keys):
    for k in keys[:-1]:
        raw = raw[k]
    del raw[keys[-1]]


@pytest.mark.parametrize(
    ("mutate", "expected_match"),
    [
        (
            lambda raw: _set(
                raw,
                "transition_model",
                "transition_probabilities",
                "sedentary",
                "nudge",
                "active",
                -0.5,
            ),
            "negative",
        ),
        (
            lambda raw: _set(
                raw,
                "transition_model",
                "transition_probabilities",
                "sedentary",
                "nudge",
                "active",
                0.5,
            ),
            "sum",
        ),
        (lambda raw: _set(raw, "initial_state", {"nonexistent": "value"}), "not found"),
        (
            lambda raw: _set(raw, "initial_state", "activity_level", "nonexistent"),
            "not in",
        ),
        (
            lambda raw: _set(
                raw,
                "reward",
                "variables",
                "bad",
                {"source": "state.nonexistent", "mapping": {"x": 0.0}},
            ),
            "unknown state variable",
        ),
        (lambda raw: _set(raw, "reward", "constants", {"value": 0.5}), "conflict"),
        (
            lambda raw: _del(raw, "transition_model", "transition_probabilities"),
            "requires either",
        ),
        (
            lambda raw: _del(
                raw,
                "transition_model",
                "transition_probabilities",
                "active",
            ),
            "missing from transition_probabilities",
        ),
        (
            lambda raw: _set(
                raw,
                "transition_model",
                "transition_probabilities",
                "sedentary",
                "nudge",
                {"nonexistent": 0.7, "active": 0.3},
            ),
            "not in",
        ),
        (
            lambda raw: _set(
                raw,
                "transition_model",
                "transition_probabilities",
                "sedentary",
                "nudge",
                {"active": 1.0},
            ),
            "cover all",
        ),
        (
            lambda raw: _del(
                raw,
                "transition_model",
                "transition_probabilities",
                "sedentary",
                "idle",
            ),
            "Missing transition",
        ),
        (lambda raw: _set(raw, "reward_multiplier_by_step", [1, 1]), "length"),
        (
            lambda raw: _set(
                raw,
                "agents",
                [{"type": "thompson_sampling", "alpha_prior": 0, "beta_prior": 1}],
            ),
            "alpha_prior",
        ),
        (
            lambda raw: _set(
                raw,
                "agents",
                [{"type": "epsilon_greedy", "epsilon": 1.5}],
            ),
            "epsilon",
        ),
        (lambda raw: _set(raw, "agents", [{"type": "ucb", "c": 0}]), "c"),
        (lambda raw: _set(raw, "agents", [{"type": "unknown_agent"}]), "unknown"),
        (
            lambda raw: _set(
                raw, "agents", [{"type": "dqn", "lr": 0.01, "batch_size": 0}]
            ),
            "batch_size",
        ),
        (lambda raw: _set(raw, "agents", [{"type": "q_learning", "lr": 0}]), "lr"),
        (
            lambda raw: _set(raw, "agents", [{"type": "q_learning"}]),
            "lr",
        ),
        (
            lambda raw: _set(raw, "agents", [{"type": "dqn"}]),
            "lr",
        ),
        (
            lambda raw: _set(raw, "agents", [{"type": "reinforce"}]),
            "lr",
        ),
        (
            lambda raw: _set(raw, "agents", [{"type": "ppo"}]),
            "lr",
        ),
        (
            lambda raw: _set(
                raw, "agents", [{"type": "reinforce", "lr": 0.01, "gamma": 0}]
            ),
            "gamma",
        ),
        (
            lambda raw: _set(
                raw, "agents", [{"type": "ppo", "lr": 0.01, "clip_eps": 0}]
            ),
            "clip_eps",
        ),
        (
            lambda raw: _set(
                raw,
                "agents",
                [
                    {
                        "type": "random",
                        "contextual": True,
                        "context_features": "activity_level",
                    }
                ],
            ),
            "contextual=True is only supported",
        ),
        (
            lambda raw: _set(
                raw,
                "agents",
                [
                    {
                        "type": "epsilon_greedy",
                        "epsilon": 0.1,
                        "context_features": "activity_level",
                    }
                ],
            ),
            "context_features must not be provided",
        ),
        (
            lambda raw: _set(
                raw,
                "agents",
                [{"type": "random", "epsilon": 0.1}],
            ),
            "does not accept any hyperparameters",
        ),
        (
            lambda raw: _set(
                raw,
                "agents",
                [{"type": "fixed", "action": "idle", "contextual": True}],
            ),
            "fixed agent does not accept learning hyperparameters or contextual",
        ),
        (
            lambda raw: _set(
                raw,
                "state",
                "variables",
                "day",
                {"names": ["monday", "tuesday"]},
            ),
            "reserved names",
        ),
    ],
    ids=[
        "negative_prob",
        "prob_sum",
        "initial_state_not_found",
        "initial_state_value_not_in_names",
        "reward_unknown_state_variable",
        "reward_constant_variable_conflict",
        "rule_based_requires_tprobs_or_table_dir",
        "tprobs_missing_state",
        "tprobs_target_not_in_names",
        "tprobs_incomplete_distribution",
        "tprobs_missing_action_entry",
        "reward_multiplier_wrong_length",
        "thompson_sampling_alpha_prior_zero",
        "epsilon_greedy_epsilon_out_of_range",
        "ucb_c_non_positive",
        "unknown_agent_type",
        "dqn_batch_size_non_positive",
        "q_learning_lr_non_positive",
        "q_learning_lr_missing",
        "dqn_lr_missing",
        "reinforce_lr_missing",
        "ppo_lr_missing",
        "reinforce_gamma_non_positive",
        "ppo_clip_eps_non_positive",
        "random_contextual_not_supported",
        "context_features_without_contextual",
        "random_rejects_hyperparameters",
        "fixed_agent_rejects_contextual",
        "reserved_state_name",
    ],
)
def test_validation_error(mutate, expected_match):
    raw = _valid_raw()
    mutate(raw)
    with pytest.raises(ValidationError, match=expected_match):
        MDPConfig.model_validate(raw)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: _set(
            raw,
            "agents",
            [
                {"type": "thompson_sampling", "alpha_prior": 1, "beta_prior": 1},
                {"type": "random"},
                {"type": "epsilon_greedy", "epsilon": 0.1},
                {"type": "ucb", "c": 2.0},
                {"type": "fixed", "action": "idle"},
            ],
        ),
        lambda raw: _set(
            raw,
            "agents",
            [
                {
                    "type": "thompson_sampling",
                    "alpha_prior": 1,
                    "beta_prior": 1,
                    "contextual": True,
                    "context_features": "activity_level",
                }
            ],
        ),
        lambda raw: _set(raw, "transition_model", {"type": "random"}),
        lambda raw: _set(raw, "agents", [{"type": "fixed", "action": "idle"}]),
        lambda raw: _set(
            raw,
            "agents",
            [
                {
                    "type": "thompson_sampling",
                    "alpha_prior": 1,
                    "beta_prior": 1,
                    "contextual": True,
                    "context_features": ["step_bin", "burden"],
                }
            ],
        ),
        lambda raw: _set(
            raw,
            "agents",
            [{"type": "q_learning", "lr": 0.05, "gamma": 0.9, "epsilon": 0.1}],
        ),
        lambda raw: _set(
            raw,
            "agents",
            [
                {
                    "type": "dqn",
                    "lr": 0.01,
                    "gamma": 0.95,
                    "epsilon": 0.2,
                    "batch_size": 8,
                    "buffer_size": 64,
                    "target_update_freq": 10,
                    "hidden_dim": [16, 8],
                    "state_dim": 32,
                }
            ],
        ),
        lambda raw: _set(
            raw,
            "agents",
            [{"type": "reinforce", "lr": 0.01, "gamma": 0.99, "hidden_dim": 16}],
        ),
        lambda raw: _set(
            raw,
            "agents",
            [
                {
                    "type": "ppo",
                    "lr": 0.01,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_eps": 0.2,
                    "ppo_epochs": 2,
                    "policy_hidden_dim": [16, 8],
                    "value_hidden_dim": [16, 8],
                    "state_dim": 32,
                }
            ],
        ),
    ],
)
def test_valid_configs(mutate):
    raw = _valid_raw()
    mutate(raw)
    MDPConfig.model_validate(raw)  # Should not raise


def test_list_actions_coerced_to_dict():
    raw = _valid_raw()
    config = MDPConfig.model_validate(raw)
    assert isinstance(config.actions, dict)
    assert "nudge" in config.actions
    assert "idle" in config.actions


def test_dict_actions_returned_directly():
    raw = _valid_raw()
    raw["actions"] = {"nudge": {}, "idle": {}}
    config = MDPConfig.model_validate(raw)
    assert config.action_names == ["nudge", "idle"]
