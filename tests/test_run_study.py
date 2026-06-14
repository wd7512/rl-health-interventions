"""Tests for full simulation runner (Task 4.1)."""

import numpy as np
import pytest

from paper_reproduction.simulation.results import (
    SimulationResults,
    save_results,
    load_results,
)
from paper_reproduction.simulation.run_study import (
    run_simulation,
    estimate_noise_variance,
    _extract_bandit_prior,
)
from paper_reproduction.data.generative_model import GenerativeModel
from paper_reproduction.data.nhanes_loader import SyntheticNHANESGenerator


class TestEstimateNoiseVariance:
    def test_returns_positive_float(self):
        step_data = SyntheticNHANESGenerator(seed=42).generate(
            n_participants=6,
            n_days=7,
            n_windows=10,
        )
        gm = GenerativeModel(
            step_data=step_data,
            alpha=np.array(
                [1.0, 0.5, 0.3, 0.1, 0.2, 0.1, 0.05, 0.05, 0.2, 0.15, 0.1, 0.05]
            ),
            beta=np.array([0.5, 0.3, 0.2, 0.1]),
            noise_variance=1.0,
            seed=42,
        )
        noise_var = estimate_noise_variance(
            training_indices=[0, 1, 2],
            generative_model=gm,
            n_days=15,
            rng=np.random.default_rng(42),
        )
        assert noise_var > 0
        assert isinstance(noise_var, float)


class TestExtractBanditPrior:
    def test_correct_shapes(self):
        g_dim, f_dim = 8, 4
        total_action = g_dim + 2 * f_dim
        prior_mean = np.arange(total_action, dtype=float)
        prior_cov = np.diag(np.arange(1, total_action + 1, dtype=float))

        bandit_mean, bandit_cov = _extract_bandit_prior(
            prior_mean,
            prior_cov,
            g_dim=g_dim,
            f_dim=f_dim,
        )

        assert bandit_mean.shape == (g_dim + f_dim,)
        assert bandit_cov.shape == (g_dim + f_dim, g_dim + f_dim)

    def test_alpha_extracted_correctly(self):
        g_dim, f_dim = 8, 4
        total = g_dim + 2 * f_dim
        prior_mean = np.arange(total, dtype=float)
        prior_cov = np.eye(total)

        bandit_mean, _ = _extract_bandit_prior(prior_mean, prior_cov, g_dim, f_dim)

        np.testing.assert_array_equal(bandit_mean[:g_dim], prior_mean[:g_dim])
        np.testing.assert_array_equal(
            bandit_mean[g_dim:],
            prior_mean[g_dim + f_dim :],
        )


class TestRunSimulation:
    def test_returns_simulation_results(self):
        results = run_simulation(
            n_participants=9,
            n_days=15,
            n_re_runs=2,
            seed=42,
        )
        assert isinstance(results, SimulationResults)

    def test_three_folds(self):
        results = run_simulation(
            n_participants=9,
            n_days=15,
            n_re_runs=2,
            seed=42,
        )
        assert len(results.cv_results) == 3

    def test_all_participants_covered(self):
        results = run_simulation(
            n_participants=9,
            n_days=15,
            n_re_runs=2,
            seed=42,
        )
        all_participants: set[int] = set()
        for cv in results.cv_results:
            for pr in cv.participant_results:
                all_participants.add(pr.participant_idx)
        assert len(all_participants) == 9

    def test_each_participant_once(self):
        results = run_simulation(
            n_participants=9,
            n_days=15,
            n_re_runs=2,
            seed=42,
        )
        all_parts: list[int] = []
        for cv in results.cv_results:
            for pr in cv.participant_results:
                all_parts.append(pr.participant_idx)
        assert len(all_parts) == 9
        assert len(set(all_parts)) == 9

    def test_summary_has_expected_keys(self):
        results = run_simulation(
            n_participants=9,
            n_days=15,
            n_re_runs=2,
            seed=42,
        )
        summary = results.compute_summary()
        expected_keys = {
            "n_participants",
            "n_improved",
            "pct_improved",
            "mean_improvement",
            "median_improvement",
            "std_improvement",
            "min_improvement",
            "max_improvement",
        }
        assert expected_keys.issubset(summary.keys())

    def test_improvement_is_float(self):
        results = run_simulation(
            n_participants=9,
            n_days=15,
            n_re_runs=2,
            seed=42,
        )
        for cv in results.cv_results:
            for pr in cv.participant_results:
                assert isinstance(pr.improvement, float)

    def test_deterministic_with_same_seed(self):
        r1 = run_simulation(
            n_participants=9,
            n_days=15,
            n_re_runs=2,
            seed=42,
        )
        r2 = run_simulation(
            n_participants=9,
            n_days=15,
            n_re_runs=2,
            seed=42,
        )
        s1 = r1.compute_summary()
        s2 = r2.compute_summary()
        assert s1["mean_improvement"] == pytest.approx(s2["mean_improvement"])

    def test_config_present(self):
        results = run_simulation(
            n_participants=9,
            n_days=15,
            n_re_runs=2,
            seed=42,
        )
        assert "n_participants" in results.config
        assert results.config["n_participants"] == 9

    def test_cv_results_have_best_params(self):
        results = run_simulation(
            n_participants=9,
            n_days=15,
            n_re_runs=2,
            seed=42,
        )
        for cv in results.cv_results:
            assert isinstance(cv.best_gamma, (int, float))
            assert isinstance(cv.best_w, (int, float))

    def test_every_participant_has_rewards(self):
        results = run_simulation(
            n_participants=9,
            n_days=15,
            n_re_runs=2,
            seed=42,
        )
        for cv in results.cv_results:
            for pr in cv.participant_results:
                assert len(pr.proposed_rewards) == 2
                assert len(pr.ts_bandit_rewards) == 2


class TestResultsSerialization:
    def test_save_and_load(self, tmp_path):
        results = run_simulation(
            n_participants=9,
            n_days=15,
            n_re_runs=2,
            seed=42,
        )
        path = str(tmp_path / "results.json")
        save_results(results, path)

        loaded = load_results(path)
        assert "config" in loaded
        assert "cv_results" in loaded
        assert "summary" in loaded
        assert len(loaded["cv_results"]) == 3

    def test_loaded_summary_matches(self, tmp_path):
        results = run_simulation(
            n_participants=9,
            n_days=15,
            n_re_runs=2,
            seed=42,
        )
        path = str(tmp_path / "results.json")
        save_results(results, path)

        loaded = load_results(path)
        original = results.compute_summary()
        for key in original:
            assert key in loaded["summary"]
            if isinstance(original[key], float):
                assert loaded["summary"][key] == pytest.approx(original[key])
            else:
                assert loaded["summary"][key] == original[key]


class TestPipelineNoErrors:
    def test_runs_with_minimal_params(self):
        results = run_simulation(
            n_participants=6,
            n_days=10,
            n_re_runs=1,
            seed=42,
        )
        assert results.compute_summary()["n_participants"] == 6

    def test_runs_with_different_gamma_w(self):
        results = run_simulation(
            n_participants=6,
            n_days=10,
            n_re_runs=1,
            gamma_values=(0.5, 0.9),
            w_values=(0.5, 1.0),
            seed=42,
        )
        assert results.compute_summary()["n_participants"] == 6
