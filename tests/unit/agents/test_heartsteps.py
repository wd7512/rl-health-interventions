from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rl_health_interventions.agents.heartsteps.agent import HeartStepsAgent
from rl_health_interventions.agents.heartsteps.bayesian_regression import (
    MultiClassBayesianRegression,
)
from rl_health_interventions.agents.heartsteps.dosage import DosageTracker
from rl_health_interventions.agents.heartsteps.proxy_value import ProxyValue
from rl_health_interventions.state import StateView


class TestDosageTracker:
    def test_initial_dosage_is_zero(self) -> None:
        assert DosageTracker().get_dosage() == 0.0

    def test_dosage_increases_on_suggestion(self) -> None:
        tracker = DosageTracker(lambda_decay=0.95)
        tracker.update(suggestion_sent=True)
        assert tracker.get_dosage() == pytest.approx(1.0)

    def test_dosage_decays_on_no_suggestion(self) -> None:
        tracker = DosageTracker(lambda_decay=0.95)
        tracker.update(suggestion_sent=True)
        tracker.update(suggestion_sent=False)
        assert tracker.get_dosage() == pytest.approx(0.95)

    def test_dosage_accumulates(self) -> None:
        tracker = DosageTracker(lambda_decay=0.9)
        tracker.update(suggestion_sent=True)
        tracker.update(suggestion_sent=True)
        assert tracker.get_dosage() == pytest.approx(1.9)

    def test_reset(self) -> None:
        tracker = DosageTracker()
        tracker.update(suggestion_sent=True)
        tracker.reset()
        assert tracker.get_dosage() == 0.0

    def test_invalid_lambda_raises(self) -> None:
        with pytest.raises(ValueError, match="lambda_decay"):
            DosageTracker(lambda_decay=0.0)
        with pytest.raises(ValueError, match="lambda_decay"):
            DosageTracker(lambda_decay=1.0)


class TestBayesianRegression:
    def test_prior_shape(self) -> None:
        reg = MultiClassBayesianRegression(
            n_features=3, actions=["a", "b", "c"], reference_action="a"
        )
        means = reg.get_beta_means()
        assert set(means.keys()) == {"a", "b", "c"}
        assert means["a"].shape == (3,)
        assert np.all(means["a"] == 0.0)

    def test_update_moves_posterior(self) -> None:
        reg = MultiClassBayesianRegression(
            n_features=2, actions=["idle", "nudge"], reference_action="idle"
        )
        reg.update_batch([(np.array([1.0, 0.0]), "nudge", 5.0)])
        assert reg.get_beta_means()["nudge"][0] != 0.0

    def test_reference_action_mean_is_zero(self) -> None:
        reg = MultiClassBayesianRegression(
            n_features=2, actions=["idle", "nudge"], reference_action="idle"
        )
        reg.update_batch([(np.array([1.0, 0.0]), "nudge", 5.0)])
        assert np.all(reg.get_beta_means()["idle"] == 0.0)

    def test_sample_betas_shape(self) -> None:
        rng = np.random.default_rng(42)
        reg = MultiClassBayesianRegression(
            n_features=3, actions=["a", "b"], reference_action="a"
        )
        samples = reg.sample_betas(rng)
        assert samples["a"].shape == (3,)
        assert samples["b"].shape == (3,)

    def test_get_reward_samples_returns_all_actions(self) -> None:
        rng = np.random.default_rng(42)
        reg = MultiClassBayesianRegression(
            n_features=2, actions=["idle", "nudge"], reference_action="idle"
        )
        reg.update_batch([(np.array([1.0, 0.0]), "nudge", 5.0)])
        f = np.array([0.5, 0.5])
        samples = reg.get_reward_samples(rng, avg_features=f)
        assert set(samples.keys()) == {"idle", "nudge"}
        assert isinstance(samples["idle"], float)
        assert isinstance(samples["nudge"], float)

    def test_batch_matches_sequential(self) -> None:
        reg_batch = MultiClassBayesianRegression(
            n_features=2, actions=["idle", "nudge"], reference_action="idle"
        )
        reg_seq = MultiClassBayesianRegression(
            n_features=2, actions=["idle", "nudge"], reference_action="idle"
        )
        transitions = [
            (np.array([1.0, 0.0]), "nudge", 3.0),
            (np.array([0.0, 1.0]), "nudge", 1.0),
            (np.array([1.0, 1.0]), "idle", 0.0),
        ]
        reg_batch.update_batch(transitions)
        for t in transitions:
            reg_seq.update_batch([t])
        assert np.allclose(
            reg_batch.get_beta_means()["nudge"],
            reg_seq.get_beta_means()["nudge"],
        )

    def test_invalid_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma_sq"):
            MultiClassBayesianRegression(
                n_features=2, actions=["a"], reference_action="a", sigma_sq=0
            )


class TestProxyValue:
    def test_initial_h_is_zero(self) -> None:
        proxy = ProxyValue(actions=["idle", "nudge"], reference_action="idle")
        assert np.all(proxy.value_table == 0.0)

    def test_eta_is_zero_initially(self) -> None:
        proxy = ProxyValue(actions=["idle", "nudge"], reference_action="idle")
        assert proxy.get_eta("nudge", 0.0) == pytest.approx(0.0)
        assert proxy.get_eta("idle", 0.0) == pytest.approx(0.0)

    def test_update_changes_h(self) -> None:
        proxy = ProxyValue(actions=["idle", "nudge"], reference_action="idle", w=1.0)
        proxy.update({"idle": 0.0, "nudge": 1.0})
        assert not np.all(proxy.value_table == 0.0)

    def test_eta_positive_for_lower_value_action(self) -> None:
        proxy = ProxyValue(
            actions=["idle", "nudge"],
            reference_action="idle",
            w=1.0,
            gamma=0.99,
        )
        for _ in range(50):
            proxy.update({"idle": 1.0, "nudge": 0.5})
        eta = proxy.get_eta("nudge", 1.0)
        assert eta > 0

    def test_custom_suggestion_actions(self) -> None:
        proxy = ProxyValue(
            actions=["idle", "walk", "run"],
            reference_action="idle",
            suggestion_actions=["walk"],
        )
        assert proxy._is_suggestion["idle"] is False
        assert proxy._is_suggestion["walk"] is True
        assert proxy._is_suggestion["run"] is False


def _make_agent(actions: list[str] | None = None, **kwargs: object) -> HeartStepsAgent:
    actions = actions or ["idle", "nudge"]
    agent = HeartStepsAgent(actions=actions, **kwargs)
    agent.init_one_hot_map({"activity_level": ["sedentary", "active"]})
    return agent


class TestHeartStepsAgent:
    def test_select_action_returns_valid(self) -> None:
        agent = _make_agent()
        state = StateView({"activity_level": "sedentary"})
        assert agent.select_action(state) in ["idle", "nudge"]

    def test_deterministic_with_same_seed(self) -> None:
        a1 = _make_agent(seed=123)
        a2 = _make_agent(seed=123)
        state = StateView({"activity_level": "sedentary"})
        acts1 = [a1.select_action(state) for _ in range(20)]
        acts2 = [a2.select_action(state) for _ in range(20)]
        assert acts1 == acts2

    def test_update_buffers_transition(self) -> None:
        agent = _make_agent()
        s = StateView({"activity_level": "sedentary"})
        ns = StateView({"activity_level": "active"})
        agent.update(s, "nudge", 1.0, ns)
        assert len(agent._buffer) == 1

    def test_on_day_end_clears_buffer(self) -> None:
        agent = _make_agent()
        s = StateView({"activity_level": "sedentary"})
        ns = StateView({"activity_level": "active"})
        agent.update(s, "nudge", 1.0, ns)
        agent.on_day_end()
        assert len(agent._buffer) == 0

    def test_on_day_end_noop_when_empty(self) -> None:
        agent = _make_agent()
        agent.on_day_end()
        assert len(agent._buffer) == 0

    def test_multi_class_actions(self) -> None:
        actions = ["idle", "movement_suggestion", "goal_reminder", "journal"]
        agent = _make_agent(actions=actions)
        state = StateView({"activity_level": "sedentary"})
        assert agent.select_action(state) in actions

    def test_thompson_samples_exploration(self) -> None:
        """TS selects different actions across seeds when posterior has variance."""
        state = StateView({"activity_level": "sedentary"})
        actions_by_seed = set()
        for seed in range(20):
            agent = _make_agent(seed=seed)
            actions_by_seed.add(agent.select_action(state))
        assert len(actions_by_seed) > 1

    def test_greedy_select_action(self) -> None:
        agent = _make_agent(greedy=True)
        state = StateView({"activity_level": "sedentary"})
        # Greedy should still return a valid action
        assert agent.select_action(state) in ["idle", "nudge"]

    def test_greedy_deterministic(self) -> None:
        a1 = _make_agent(seed=123, greedy=True)
        a2 = _make_agent(seed=123, greedy=True)
        state = StateView({"activity_level": "sedentary"})
        assert a1.select_action(state) == a2.select_action(state)

    def test_no_proxy(self) -> None:
        agent = _make_agent(use_proxy=False)
        state = StateView({"activity_level": "sedentary"})
        assert agent.select_action(state) in ["idle", "nudge"]

    def test_works_with_multiple_state_factors(self) -> None:
        agent = HeartStepsAgent(actions=["idle", "nudge"], seed=42)
        agent.init_one_hot_map(
            {"step_bin": ["inactive", "active"], "sleep": ["good", "poor"]}
        )
        state = StateView({"step_bin": "inactive", "sleep": "good"})
        assert agent.select_action(state) in ["idle", "nudge"]

    def test_one_hot_encoding(self) -> None:
        agent = _make_agent()
        vec = agent._one_hot(StateView({"activity_level": "active"}))
        assert vec.shape == (2,)
        assert vec[0] == 0.0
        assert vec[1] == 1.0

    def test_one_hot_encoding_multi_factor(self) -> None:
        agent = HeartStepsAgent(actions=["idle", "nudge"], seed=42)
        agent.init_one_hot_map(
            {"step_bin": ["inactive", "active"], "sleep": ["good", "poor"]}
        )
        vec = agent._one_hot(StateView({"step_bin": "active", "sleep": "poor"}))
        assert vec.shape == (4,)
        # step_bin_active=1, sleep_poor=1
        assert vec[1] == 1.0
        assert vec[3] == 1.0


class TestHeartStepsIntegration:
    def test_runs_episode_with_rule_based(self) -> None:
        from rl_health_interventions.agents import make
        from rl_health_interventions.config.loader import load_config
        from rl_health_interventions.episode import run_episode

        config_path = (
            Path(__file__).parents[3]
            / "docs"
            / "experiments"
            / "mvp"
            / "configs"
            / "mvp.yaml"
        )
        config = load_config(config_path)
        agent = make("heartsteps", actions=config.action_names, seed=42)
        state_variables = {
            name: cfg.names for name, cfg in config.state.variables.items()
        }
        extra = {"step_of_day": list(range(config.steps_per_day))}
        agent.init_one_hot_map(state_variables, extra_features=extra)
        records = run_episode(config, agent, seed=42)
        assert len(records) == config.episode_days * config.steps_per_day
        assert all(r["action"] in config.action_names for r in records)

    def test_on_day_end_fires_at_correct_count(self) -> None:
        """on_day_end is called once per day boundary: episode_days - 1 times."""
        from rl_health_interventions.config.loader import load_config
        from rl_health_interventions.episode import run_episode

        config_path = (
            Path(__file__).parents[3]
            / "docs"
            / "experiments"
            / "mvp"
            / "configs"
            / "mvp.yaml"
        )
        config = load_config(config_path)
        agent = _make_agent(actions=config.action_names)
        state_variables = {
            name: cfg.names for name, cfg in config.state.variables.items()
        }
        extra = {"step_of_day": list(range(config.steps_per_day))}
        agent.init_one_hot_map(state_variables, extra_features=extra)
        # Track on_day_end calls
        original_on_day_end = agent.on_day_end
        call_count = 0

        def counting_on_day_end() -> None:
            nonlocal call_count
            call_count += 1
            original_on_day_end()

        agent.on_day_end = counting_on_day_end  # type: ignore[assignment]
        run_episode(config, agent, seed=42)
        assert call_count == config.episode_days
