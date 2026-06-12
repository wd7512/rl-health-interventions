"""Full simulation runner for HeartSteps V2 reproduction study.

Wires together data generation, cross-validation, prior construction,
noise variance estimation, tuning parameter grid search, and evaluation
of the proposed algorithm against the TS Bandit baseline.

Reference:
    Liao et al. (2019). arXiv:1909.03539, Section 6.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from paper_reproduction.baselines.ts_bandit import TSBandit
from paper_reproduction.data.generative_model import GenerativeModel
from paper_reproduction.data.nhanes_loader import NHANESLoader, SyntheticNHANESGenerator
from paper_reproduction.heartsteps.agent import ThompsonSamplingAgent
from paper_reproduction.heartsteps.bayesian_regression import BayesianRewardModel
from paper_reproduction.heartsteps.dosage import DosageTracker
from paper_reproduction.heartsteps.proxy_value import ProxyValueFunction
from paper_reproduction.simulation.cross_validation import create_folds
from paper_reproduction.simulation.prior_construction import construct_prior
from paper_reproduction.simulation.results import (
    CVFoldResult,
    ParticipantResult,
    SimulationResults,
)
from paper_reproduction.simulation.tuning import grid_search

logger = logging.getLogger(__name__)


def estimate_noise_variance(
    training_indices: list[int],
    generative_model: GenerativeModel,
    n_days: int = 90,
    pi_param: float = 0.3,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate observation noise variance from training batch OLS residuals.

    Fits the action-centered model on pooled training data and returns the
    residual variance as an estimate of sigma^2.

    Args:
        training_indices: Participant indices in the training batch.
        generative_model: Generative model for simulation.
        n_days: Days per trajectory.
        pi_param: Randomization probability.
        rng: Random generator.

    Returns:
        Estimated noise variance (scalar, clamped to >= 0.01).
    """
    from paper_reproduction.simulation.prior_construction import _extract_observations

    if rng is None:
        rng = np.random.default_rng()

    all_residuals: list[float] = []

    for idx in training_indices:
        X, y = _extract_observations(generative_model, idx, n_days, pi_param)
        if X.shape[0] <= X.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        all_residuals.extend(float(r) for r in residuals)

    if not all_residuals:
        logger.warning("No training data for noise variance; defaulting to 1.0")
        return 1.0

    return max(float(np.var(all_residuals, ddof=1)), 0.01)


def _simulate_episode_proposed(
    generative_model: GenerativeModel,
    agent: ThompsonSamplingAgent,
    participant_idx: int,
    n_days: int = 90,
    pi_param: float = 0.3,
    rng: np.random.Generator | None = None,
) -> float:
    """Run one interactive simulation episode with the proposed TS agent.

    Steps through decision times: agent selects action, reward is
    computed from the generative model, and the model updates its posterior.

    Args:
        generative_model: Model providing step data and reward structure.
        agent: TS agent with proxy value and dosage tracking.
        participant_idx: Index of the participant to simulate.
        n_days: Number of days to simulate.
        pi_param: Fixed randomization probability.
        rng: Random generator.

    Returns:
        Total reward accumulated over the episode.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_windows = generative_model.step_data.shape[2]
    total_times = n_days * n_windows
    data = generative_model._extend_to_90_days(participant_idx)
    steps = data.flatten()

    total_reward = 0.0
    daily_steps = np.zeros(total_times, dtype=float)

    for t in range(total_times):
        day = t // n_windows
        time_slot = t % n_windows // 2
        window = t % n_windows

        is_night = window < 1 or window >= 9
        available = (
            False if is_night else rng.binomial(1, generative_model.p_avail) == 1
        )

        if time_slot == 0:
            daily_steps[t] = steps[max(0, t - n_windows) : t].sum()
        else:
            daily_steps[t] = daily_steps[max(t - n_windows, 0)]

        g, f = generative_model._construct_features(
            steps,
            t,
            daily_steps[t],
            time_slot,
            day,
        )

        action, _ = agent.select_action(g, f, pi=pi_param, available=available)

        # Anti-sedentary suggestion (external to RL, probability p_sed)
        anti_sedentary = rng.binomial(1, generative_model.p_sed) == 1

        alpha_0 = generative_model.alpha[: generative_model.g_dim]
        alpha_1 = generative_model.alpha[generative_model.g_dim :]
        R_base = float(g @ alpha_0 + f @ alpha_1)
        f_beta = f[: len(generative_model.beta)]
        R_treat = float(action * f_beta @ generative_model.beta)
        epsilon = float(rng.normal(0, np.sqrt(generative_model.noise_variance)))
        reward = R_base + R_treat + epsilon

        agent.step(
            g,
            f,
            action,
            pi=pi_param,
            reward=reward,
            available=available,
            anti_sedentary=anti_sedentary,
        )
        total_reward += reward

    return total_reward


def _simulate_episode_ts_bandit(
    generative_model: GenerativeModel,
    bandit: TSBandit,
    participant_idx: int,
    n_days: int = 90,
    pi_param: float = 0.3,
    rng: np.random.Generator | None = None,
) -> float:
    """Run one interactive simulation episode with the TS Bandit baseline.

    Args:
        generative_model: Model providing step data and reward structure.
        bandit: TS Bandit (no proxy value, no action-centering).
        participant_idx: Index of the participant to simulate.
        n_days: Days to simulate.
        pi_param: Randomization probability (passed but unused by bandit).
        rng: Random generator.

    Returns:
        Total reward accumulated over the episode.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_windows = generative_model.step_data.shape[2]
    total_times = n_days * n_windows
    data = generative_model._extend_to_90_days(participant_idx)
    steps = data.flatten()

    total_reward = 0.0
    daily_steps = np.zeros(total_times, dtype=float)

    for t in range(total_times):
        day = t // n_windows
        time_slot = t % n_windows // 2
        window = t % n_windows

        is_night = window < 1 or window >= 9
        available = (
            False if is_night else rng.binomial(1, generative_model.p_avail) == 1
        )

        if time_slot == 0:
            daily_steps[t] = steps[max(0, t - n_windows) : t].sum()
        else:
            daily_steps[t] = daily_steps[max(t - n_windows, 0)]

        g, f = generative_model._construct_features(
            steps,
            t,
            daily_steps[t],
            time_slot,
            day,
        )

        action, _ = bandit.select_action(g, f, available=available)

        alpha_0 = generative_model.alpha[: generative_model.g_dim]
        alpha_1 = generative_model.alpha[generative_model.g_dim :]
        R_base = float(g @ alpha_0 + f @ alpha_1)
        f_beta = f[: len(generative_model.beta)]
        R_treat = float(action * f_beta @ generative_model.beta)
        epsilon = float(rng.normal(0, np.sqrt(generative_model.noise_variance)))
        reward = R_base + R_treat + epsilon

        bandit.step(g, f, action, pi=pi_param, reward=reward, available=available)
        total_reward += reward

    return total_reward


def _extract_bandit_prior(
    prior_mean_action_centered: np.ndarray,
    prior_cov_action_centered: np.ndarray,
    g_dim: int = 8,
    f_dim: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract TS Bandit prior from action-centered prior.

    The action-centered parameter vector is [alpha_0, alpha_1, beta].
    The TS Bandit vector is [alpha, beta] where alpha = alpha_0.

    Args:
        prior_mean_action_centered: Mean from action-centered model,
            shape (g_dim + 2*f_dim,).
        prior_cov_action_centered: Covariance from action-centered model,
            shape (g_dim + 2*f_dim, g_dim + 2*f_dim).
        g_dim: Baseline feature dimension.
        f_dim: Treatment feature dimension.

    Returns:
        Tuple of (bandit_prior_mean, bandit_prior_cov).
    """
    idx = list(range(g_dim)) + list(range(g_dim + f_dim, g_dim + 2 * f_dim))
    bandit_mean = prior_mean_action_centered[idx]
    bandit_cov = prior_cov_action_centered[np.ix_(idx, idx)]
    return bandit_mean, bandit_cov


def run_simulation(
    n_participants: int = 30,
    n_folds: int = 3,
    n_source_days: int = 7,
    n_days: int = 90,
    n_windows: int = 10,
    n_re_runs: int = 10,
    gamma_values: tuple[float, ...] = (0, 0.25, 0.5, 0.75, 0.9, 0.95),
    w_values: tuple[float, ...] = (0, 0.1, 0.25, 0.5, 0.75, 1.0),
    seed: int = 42,
    data_source: str = "synthetic",
    data_path: str | None = None,
) -> SimulationResults:
    """Run the full simulation study end-to-end.

    Steps:
    1. Generate synthetic NHANES-like step data.
    2. Partition participants into 3 CV folds.
    3. For each fold:
       a. Construct prior from training batch (action-centered).
       b. Estimate noise variance from training batch OLS residuals.
       c. Grid-search (gamma, w) on training batch.
       d. For each test participant, run proposed algorithm and TS Bandit
          n_re_runs times each, recording average total reward.
    4. Aggregate results into SimulationResults.

    Args:
        n_participants: Total number of participants to simulate.
        n_folds: Number of cross-validation folds.
        n_source_days: Days of source data per participant.
        n_days: Days per simulation episode.
        n_windows: Decision windows per day.
        n_re_runs: Re-runs per participant per algorithm.
        gamma_values: Grid values for discount rate gamma.
        w_values: Grid values for blending weight w.
        seed: Top-level random seed.

    Returns:
        SimulationResults with all fold-level and participant-level data.
    """
    rng = np.random.default_rng(seed)

    logger.info(
        "Starting simulation: %d participants, %d folds, %d days, "
        "%d re-runs, gamma=%s, w=%s",
        n_participants,
        n_folds,
        n_days,
        n_re_runs,
        gamma_values,
        w_values,
    )

    # --- Step 1: Load step data ---
    if data_source == "real" and data_path is not None:
        loader = NHANESLoader(
            data_source="real",
            n_participants=n_participants,
            n_days=n_source_days,
            seed=int(rng.integers(0, 2**31)),
            data_path=data_path,
        )
        step_data = loader.load()
    else:
        step_gen = SyntheticNHANESGenerator(seed=int(rng.integers(0, 2**31)))
        step_data = step_gen.generate(
            n_participants=n_participants,
            n_days=n_source_days,
            n_windows=n_windows,
        )

    # Reward model parameters matching the enriched feature setup
    alpha = np.array([1.0, 0.5, 0.3, 0.1, 0.2, 0.1, 0.2, 0.15, 0.1, 0.05])
    beta = np.array([1.5, 2.5, 1.0, 0.5])

    # --- Step 2: Create CV folds ---
    folds = create_folds(
        n_participants=n_participants,
        n_folds=n_folds,
        rng=np.random.default_rng(int(rng.integers(0, 2**31))),
    )

    config: dict[str, Any] = {
        "n_participants": n_participants,
        "n_folds": n_folds,
        "n_source_days": n_source_days,
        "n_days": n_days,
        "n_windows": n_windows,
        "n_re_runs": n_re_runs,
        "gamma_values": list(gamma_values),
        "w_values": list(w_values),
        "seed": seed,
        "data_source": data_source,
        "alpha": alpha.tolist(),
        "beta": beta.tolist(),
    }

    cv_results_list: list[CVFoldResult] = []

    for fold_idx, (train_indices, test_indices) in enumerate(folds):
        logger.info(
            "--- Fold %d/%d: train=%s, test=%s ---",
            fold_idx + 1,
            n_folds,
            train_indices.tolist(),
            test_indices.tolist(),
        )

        # Generative model for this fold
        gm = GenerativeModel(
            step_data=step_data,
            alpha=alpha,
            beta=beta,
            g_dim=6,
            noise_variance=1.0,
            p_avail=0.85,
            p_sed=0.2,
            seed=int(rng.integers(0, 2**31)),
        )

        # Step 3a: Construct prior from training batch
        prior_mean, prior_cov = construct_prior(
            training_indices=train_indices.tolist(),
            generative_model=gm,
            g_dim=6,
            f_dim=4,
            n_days=n_days,
            pi_param=0.3,
            rng=np.random.default_rng(int(rng.integers(0, 2**31))),
        )

        # Step 3b: Estimate noise variance
        noise_var = estimate_noise_variance(
            training_indices=train_indices.tolist(),
            generative_model=gm,
            n_days=n_days,
            pi_param=0.3,
            rng=np.random.default_rng(int(rng.integers(0, 2**31))),
        )

        # Step 3c: Grid search for best (gamma, w)
        grid_result = grid_search(
            step_data=step_data,
            train_indices=train_indices,
            alpha=alpha,
            beta=beta,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            g_dim=6,
            f_dim=4,
            n_re_runs=n_re_runs,
            n_days=n_days,
            n_windows=n_windows,
            noise_variance=noise_var,
            p_avail=0.85,
            p_sed=0.2,
            tau=1.0,
            epsilon_0=0.2,
            epsilon_1=0.1,
            gamma_values=gamma_values,
            w_values=w_values,
            seed=int(rng.integers(0, 2**31)),
        )

        best_gamma = grid_result["best_gamma"]
        best_w = grid_result["best_w"]

        logger.info(
            "Fold %d: best gamma=%.2f, w=%.2f, reward=%.2f",
            fold_idx + 1,
            best_gamma,
            best_w,
            grid_result["best_reward"],
        )

        # Prior for TS Bandit (non-action-centered parameters)
        bandit_prior_mean, bandit_prior_cov = _extract_bandit_prior(
            prior_mean,
            prior_cov,
            g_dim=6,
            f_dim=4,
        )

        # Steps 3d-g: Evaluate on test participants
        participant_results: list[ParticipantResult] = []

        for p_idx in test_indices:
            p_idx_int = int(p_idx)

            proposed_rewards: list[float] = []
            ts_bandit_rewards: list[float] = []

            for run in range(n_re_runs):
                run_seed = int(rng.integers(0, 2**31))

                # --- Proposed algorithm run ---
                model_proposed = BayesianRewardModel(
                    g_dim=6,
                    f_dim=4,
                    prior_mean=prior_mean.copy(),
                    prior_cov=prior_cov.copy(),
                    noise_variance=noise_var,
                )
                dosage_proposed = DosageTracker(decay=0.95)
                proxy = ProxyValueFunction(
                    decay=0.95,
                    gamma=best_gamma,
                    p_avail=0.85,
                    p_sed=0.2,
                    w=best_w,
                    grid_max=20.0,
                    grid_step=0.5,
                    treat_benefit=2.0,
                    burden_coef=0.3,
                )
                proxy.solve()

                agent = ThompsonSamplingAgent(
                    model=model_proposed,
                    dosage_tracker=dosage_proposed,
                    proxy_value=proxy,
                    tau=1.0,
                    epsilon_0=0.2,
                    epsilon_1=0.1,
                    rng=np.random.default_rng(run_seed + 1),
                )

                rew_proposed = _simulate_episode_proposed(
                    generative_model=gm,
                    agent=agent,
                    participant_idx=p_idx_int,
                    n_days=n_days,
                    pi_param=0.3,
                    rng=np.random.default_rng(run_seed + 2),
                )
                proposed_rewards.append(rew_proposed)

                # --- TS Bandit run ---
                bandit = TSBandit(
                    g_dim=6,
                    f_dim=4,
                    prior_mean=bandit_prior_mean.copy(),
                    prior_cov=bandit_prior_cov.copy(),
                    noise_variance=noise_var,
                    tau=1.0,
                    epsilon_0=0.2,
                    epsilon_1=0.1,
                    rng=np.random.default_rng(run_seed + 3),
                )

                rew_bandit = _simulate_episode_ts_bandit(
                    generative_model=gm,
                    bandit=bandit,
                    participant_idx=p_idx_int,
                    n_days=n_days,
                    pi_param=0.3,
                    rng=np.random.default_rng(run_seed + 4),
                )
                ts_bandit_rewards.append(rew_bandit)

            prop_mean = float(np.mean(proposed_rewards)) if proposed_rewards else 0.0
            bandit_mean = (
                float(np.mean(ts_bandit_rewards)) if ts_bandit_rewards else 0.0
            )
            improvement = prop_mean - bandit_mean

            participant_results.append(
                ParticipantResult(
                    participant_idx=p_idx_int,
                    proposed_mean=prop_mean,
                    ts_bandit_mean=bandit_mean,
                    improvement=improvement,
                    proposed_rewards=proposed_rewards,
                    ts_bandit_rewards=ts_bandit_rewards,
                )
            )

            logger.info(
                "Part %d: proposed=%.2f, bandit=%.2f, impr=%.2f",
                p_idx_int,
                prop_mean,
                bandit_mean,
                improvement,
            )

        cv_results_list.append(
            CVFoldResult(
                fold=fold_idx,
                train_indices=train_indices.tolist(),
                test_indices=test_indices.tolist(),
                best_gamma=best_gamma,
                best_w=best_w,
                participant_results=participant_results,
            )
        )

    results = SimulationResults(config=config, cv_results=cv_results_list)
    summary = results.compute_summary()
    logger.info(
        "Simulation complete: %.1f%% improved, mean improvement=%.2f",
        summary["pct_improved"],
        summary["mean_improvement"],
    )

    return results
