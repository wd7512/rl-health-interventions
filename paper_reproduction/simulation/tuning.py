"""Grid search over proxy value tuning parameters (gamma, w).

Implements Section 6.1 tuning procedure: for each (gamma, w) pair, runs
the TS agent on generative model trajectories for training participants
and selects the pair that maximises average total reward.

Reference:
    Liao et al. (2019). arXiv:1909.03539, Section 6.1.
"""

from __future__ import annotations

import logging

import numpy as np

from paper_reproduction.data.generative_model import GenerativeModel
from paper_reproduction.heartsteps.agent import ThompsonSamplingAgent
from paper_reproduction.heartsteps.bayesian_regression import BayesianRewardModel
from paper_reproduction.heartsteps.dosage import DosageTracker
from paper_reproduction.heartsteps.proxy_value import ProxyValueFunction

logger = logging.getLogger(__name__)


def _simulate_episode(
    generative_model: GenerativeModel,
    agent: ThompsonSamplingAgent,
    participant_idx: int,
    n_days: int = 90,
    pi_param: float = 0.3,
    rng: np.random.Generator | None = None,
) -> float:
    """Run one interactive simulation episode.

    Steps through each decision time: agent selects an action, reward is
    computed from the generative model, and the agent updates its posterior.

    Args:
        generative_model: Model providing step data and reward structure.
        agent: Agent with a specific proxy configuration.
        participant_idx: Index of the participant to simulate.
        n_days: Number of days to simulate.
        pi_param: Fixed randomization probability for action-centering.
        rng: NumPy random generator for reward noise and availability.

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
            steps[t],
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


def _run_training_participant(
    generative_model: GenerativeModel,
    agent_params: dict,
    participant_idx: int,
    n_re_runs: int,
    n_days: int,
    pi_param: float,
    rng: np.random.Generator,
) -> float:
    """Run multiple episodes for one training participant; return avg reward.

    Args:
        generative_model: The generative model for simulation.
        agent_params: Dict with keys 'model', 'dosage_tracker',
            'proxy_value', 'tau', 'epsilon_0', 'epsilon_1'.
        participant_idx: Index of the training participant.
        n_re_runs: Number of repeated runs to average.
        n_days: Days per episode.
        pi_param: Randomization probability.
        rng: Random generator.

    Returns:
        Average total reward across re-runs.
    """
    total = 0.0
    for run in range(n_re_runs):
        run_seed = int(rng.integers(0, 2**31))

        model = BayesianRewardModel(
            g_dim=agent_params["model"].g_dim,
            f_dim=agent_params["model"].f_dim,
            prior_mean=agent_params["model"].posterior_mean.copy(),
            prior_cov=agent_params["model"].posterior_cov.copy(),
            noise_variance=agent_params["model"].noise_variance,
        )
        dosage = DosageTracker(decay=agent_params["dosage_tracker"].decay)
        proxy_kwargs = dict(
            decay=agent_params["proxy_value"].decay,
            gamma=agent_params["proxy_value"].gamma,
            p_avail=agent_params["proxy_value"].p_avail,
            p_sed=agent_params["proxy_value"].p_sed,
            w=agent_params["proxy_value"].w,
            grid_max=float(agent_params["proxy_value"].grid.max()),
            grid_step=0.5,
            treat_benefit=agent_params["proxy_value"].treat_benefit,
            burden_coef=agent_params["proxy_value"].burden_coef,
        )
        proxy = ProxyValueFunction(**proxy_kwargs)
        proxy.solve()

        agent = ThompsonSamplingAgent(
            model=model,
            dosage_tracker=dosage,
            proxy_value=proxy,
            tau=agent_params["tau"],
            epsilon_0=agent_params["epsilon_0"],
            epsilon_1=agent_params["epsilon_1"],
            rng=np.random.default_rng(run_seed + 1),
        )

        total += _simulate_episode(
            generative_model=generative_model,
            agent=agent,
            participant_idx=participant_idx,
            n_days=n_days,
            pi_param=pi_param,
            rng=np.random.default_rng(run_seed + 2),
        )

    return total / max(n_re_runs, 1)


def grid_search(
    step_data: np.ndarray,
    train_indices: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    g_dim: int = 4,
    f_dim: int = 2,
    n_re_runs: int = 5,
    n_days: int = 90,
    n_windows: int = 10,
    noise_variance: float = 1.0,
    p_avail: float = 0.85,
    p_sed: float = 0.2,
    tau: float = 1.0,
    epsilon_0: float = 0.2,
    epsilon_1: float = 0.1,
    gamma_values: tuple[float, ...] = (0, 0.25, 0.5, 0.75, 0.9, 0.95),
    w_values: tuple[float, ...] = (0, 0.1, 0.25, 0.5, 0.75, 1.0),
    seed: int = 42,
) -> dict:
    """Grid search over (gamma, w) to maximise average total reward.

    For each (gamma, w) pair, runs the TS agent on generative model
    trajectories for each training participant, repeated n_re_runs times.
    Selects the pair with the highest average total reward.

    Args:
        step_data: Source step data (n_participants, n_days, n_windows).
        train_indices: Indices of training participants.
        alpha: Reward coefficients [alpha_0 (g_dim), alpha_1 (f_dim)].
        beta: True treatment effect coefficients (f_dim).
        prior_mean: Prior mean for agent's Bayesian model.
        prior_cov: Prior covariance for agent's Bayesian model.
        g_dim: Baseline feature dimension.
        f_dim: Treatment feature dimension.
        n_re_runs: Re-runs per participant per parameter pair.
        n_days: Days per simulation episode.
        n_windows: Decision windows per day.
        noise_variance: Observation noise variance.
        p_avail: Availability probability for generative model.
        p_sed: Anti-sedentary suggestion probability.
        tau: Softmax temperature for action selection.
        epsilon_0: Upper clip for action probability.
        epsilon_1: Lower clip for action probability.
        gamma_values: Grid values for discount rate gamma.
        w_values: Grid values for blending weight w.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys 'best_gamma', 'best_w', 'best_reward',
        'grid_results' (dict mapping (gamma, w) -> avg reward), and
        metadata.
    """
    rng = np.random.default_rng(seed)

    generative_model = GenerativeModel(
        step_data=step_data,
        alpha=alpha,
        beta=beta,
        g_dim=g_dim,
        noise_variance=noise_variance,
        p_avail=p_avail,
        p_sed=p_sed,
        seed=int(rng.integers(0, 2**31)),
    )

    base_model = BayesianRewardModel(
        g_dim=g_dim,
        f_dim=f_dim,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        noise_variance=noise_variance,
    )
    base_dosage = DosageTracker(decay=0.95)

    grid_results: dict[tuple[float, float], float] = {}

    best_reward = -float("inf")
    best_gamma = gamma_values[0]
    best_w = w_values[0]

    for gamma in gamma_values:
        for w_val in w_values:
            proxy = ProxyValueFunction(
                decay=0.95,
                gamma=gamma,
                p_avail=p_avail,
                p_sed=p_sed,
                w=w_val,
                grid_max=20.0,
                grid_step=0.5,
                treat_benefit=2.0,
                burden_coef=0.3,
            )
            proxy.solve()

            agent_params = {
                "model": base_model,
                "dosage_tracker": base_dosage,
                "proxy_value": proxy,
                "tau": tau,
                "epsilon_0": epsilon_0,
                "epsilon_1": epsilon_1,
            }

            participant_rewards: list[float] = []
            for p_idx in train_indices:
                p_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
                avg = _run_training_participant(
                    generative_model=generative_model,
                    agent_params=agent_params,
                    participant_idx=int(p_idx),
                    n_re_runs=n_re_runs,
                    n_days=n_days,
                    pi_param=0.3,
                    rng=p_rng,
                )
                participant_rewards.append(avg)

            mean_reward = (
                float(np.mean(participant_rewards)) if participant_rewards else 0.0
            )
            grid_results[(gamma, w_val)] = mean_reward

            if mean_reward > best_reward:
                best_reward = mean_reward
                best_gamma = gamma
                best_w = w_val

            logger.debug(
                "gamma=%.2f, w=%.2f -> avg_reward=%.2f",
                gamma,
                w_val,
                mean_reward,
            )

    return {
        "best_gamma": best_gamma,
        "best_w": best_w,
        "best_reward": best_reward,
        "grid_results": grid_results,
        "n_runs": n_re_runs,
        "n_participants": len(train_indices),
    }
