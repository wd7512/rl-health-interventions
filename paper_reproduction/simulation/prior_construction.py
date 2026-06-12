"""Construct informative priors from training data.

Implements Section 5.5 of the HeartSteps V2 paper.

Procedure:
1. Generate trajectories for training participants using the generative model.
2. Compute baseline (g) and treatment (f) features for each observation.
3. Fit population OLS (as GEE approximation) on pooled data.
4. Fit per-participant OLS to get individual coefficient estimates.
5. For each parameter, assess significance via OLS p-value.
   - Significant (p < alpha): prior mean = pooled estimate, prior std = cross-participant std.
   - Non-significant: prior mean = 0, prior std = half of cross-participant std.
   - New features: prior mean = 0, prior std = average of other features' prior stds.
6. Construct diagonal prior covariance matrix.

Reference:
    Liao et al. (2019). "Personalized HeartSteps." arXiv:1909.03539, Section 5.5.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from paper_reproduction.data.generative_model import GenerativeModel

logger = logging.getLogger(__name__)


def _compute_daily_steps(
    steps: np.ndarray, n_windows: int, total_times: int
) -> np.ndarray:
    """Compute daily step accumulation matching the generative model.

    Args:
        steps: 1-D array of step counts across all time points.
        n_windows: Number of windows per day.
        total_times: Total number of time points.

    Returns:
        Array of daily step estimates (same length as steps).
    """
    daily = np.zeros(total_times)
    for t in range(total_times):
        time_slot = t % n_windows // 2
        if time_slot == 0:
            daily[t] = steps[max(0, t - n_windows) : t].sum()
        else:
            daily[t] = daily[max(t - n_windows, 0)]
    return daily


def _construct_features(
    steps_t: float, daily_steps_t: float, time_slot: int, day: int
) -> tuple[np.ndarray, np.ndarray]:
    """Construct baseline (g) and treatment (f) feature vectors.

    Matches the construction logic in GenerativeModel._construct_features.

    Args:
        steps_t: Step count in the current 30-minute window.
        daily_steps_t: Cumulative daily steps estimate.
        time_slot: Decision time slot (0-4).
        day: Study day (0-89).

    Returns:
        Tuple of (g_features, f_features) as numpy arrays.
    """
    steps_norm = min(steps_t / 500.0, 1.0)
    daily_norm = min(daily_steps_t / 10000.0, 1.0)
    time_norm = time_slot / 4.0
    day_norm = day / 89.0
    g = np.array([steps_norm, daily_norm, time_norm, day_norm])
    step_variation = 0.5
    f = np.array([time_norm, step_variation])
    return g, f


def _fit_ols(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit ordinary least squares regression.

    Fits y = X @ beta + epsilon without an intercept (the model already
    includes baseline features in X).

    Args:
        X: Design matrix of shape (n_obs, n_features).
        y: Target vector of shape (n_obs,).

    Returns:
        Tuple of (coefficients, standard_errors, p_values).
    """
    n, p = X.shape
    if n <= p:
        return np.zeros(p), np.zeros(p), np.ones(p)

    try:
        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)

        if rank < p:
            beta = np.linalg.pinv(X) @ y

        residuals = y - X @ beta
        sigma2 = np.sum(residuals**2) / max(n - p, 1)

        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)

        se = np.sqrt(sigma2 * np.abs(np.diag(XtX_inv)))
        z_stats = np.where(se > 1e-15, beta / se, 0.0)
        p_values = np.array([
            2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
            for z in z_stats
        ])
        p_values = np.clip(p_values, 0.0, 1.0)

        return beta, se, p_values
    except np.linalg.LinAlgError:
        return np.zeros(p), np.zeros(p), np.ones(p)


def _extract_observations(
    generative_model: GenerativeModel,
    participant_idx: int,
    n_days: int,
    pi_param: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and reward vector y for one participant.

    Each observation contributes a joint feature vector
    phi = [g, pi*f, (A-pi)*f] and the observed reward R.

    Args:
        generative_model: The generative model to simulate from.
        participant_idx: Participant index.
        n_days: Number of days to simulate.
        pi_param: Randomization probability for action-centering.

    Returns:
        Tuple of (X, y) where X has shape (n_obs, g_dim + 2*f_dim).
    """
    traj = generative_model.generate_trajectory(participant_idx, n_days)
    n_windows = generative_model.step_data.shape[2]
    total_times = n_days * n_windows

    g_dim = generative_model.g_dim
    f_dim = len(generative_model.beta)
    total_params = g_dim + 2 * f_dim

    daily_steps = _compute_daily_steps(traj["steps"], n_windows, total_times)

    X_list: list[np.ndarray] = []
    y_list: list[float] = []

    for t in range(total_times):
        available = traj["availabilities"][t]
        if not available:
            continue

        day = t // n_windows
        time_slot = t % n_windows // 2

        g, f = _construct_features(
            float(traj["steps"][t]),
            float(daily_steps[t]),
            time_slot,
            day,
        )

        action = int(traj["actions"][t])
        reward = float(traj["rewards"][t])

        phi = np.concatenate([g, pi_param * f, (action - pi_param) * f])
        X_list.append(phi)
        y_list.append(reward)

    if not X_list:
        return np.zeros((0, total_params)), np.array([])

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


def construct_prior(
    training_indices: list[int],
    generative_model: GenerativeModel,
    g_dim: int = 4,
    f_dim: int = 2,
    n_days: int = 90,
    pi_param: float = 0.3,
    alpha_level: float = 0.05,
    new_feature_indices: list[int] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct informative prior from training batch.

    For each parameter in theta = [alpha_0, alpha_1, beta]:
    1. If significant (p < alpha_level):
       - Prior mean = pooled regression estimate
       - Prior std = standard deviation of per-participant estimates
    2. If non-significant:
       - Prior mean = 0, prior std = half of cross-participant std
    3. For new features (not present in generative model):
       - Prior mean = 0
       - Prior std = average of other features' prior stds

    The prior covariance matrix is diagonal.

    Args:
        training_indices: List of participant indices in the training batch.
        generative_model: Generative model for simulating trajectories.
        g_dim: Dimension of baseline features g(s). Default 4.
        f_dim: Dimension of treatment features f(s). Default 2.
        n_days: Number of simulation days per participant. Default 90.
        pi_param: Randomization probability for action-centering. Default 0.3.
        alpha_level: Significance threshold for p-values. Default 0.05.
        new_feature_indices: Indices of parameters not in the generative model.
        rng: NumPy random generator.

    Returns:
        Tuple of (prior_mean, prior_cov).
    """
    if rng is None:
        rng = np.random.default_rng()

    total_params = g_dim + 2 * f_dim

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    per_participant_X: dict[int, np.ndarray] = {}
    per_participant_y: dict[int, np.ndarray] = {}

    for idx in training_indices:
        X_i, y_i = _extract_observations(
            generative_model, idx, n_days, pi_param
        )
        all_X.append(X_i)
        all_y.append(y_i)
        per_participant_X[idx] = X_i
        per_participant_y[idx] = y_i

    if not all_X or all(X.shape[0] == 0 for X in all_X):
        logger.warning("No observations extracted; returning flat prior")
        return np.zeros(total_params), np.eye(total_params)

    X_pooled = np.vstack(all_X)
    y_pooled = np.concatenate(all_y)
    pooled_beta, pooled_se, pooled_p = _fit_ols(X_pooled, y_pooled)

    per_participant_betas = []
    for idx in training_indices:
        X_i = per_participant_X[idx]
        y_i = per_participant_y[idx]
        if X_i.shape[0] <= total_params:
            continue
        beta_i, _, _ = _fit_ols(X_i, y_i)
        per_participant_betas.append(beta_i)

    if len(per_participant_betas) >= 2:
        beta_array = np.array(per_participant_betas)
        cross_std = np.std(beta_array, axis=0, ddof=1)
    elif len(per_participant_betas) == 1:
        cross_std = pooled_se.copy()
    else:
        cross_std = np.ones(total_params)

    prior_mean = np.zeros(total_params)
    prior_std = np.zeros(total_params)

    for k in range(total_params):
        if new_feature_indices and k in new_feature_indices:
            prior_mean[k] = 0.0
            prior_std[k] = np.nan
        elif pooled_p[k] < alpha_level:
            prior_mean[k] = pooled_beta[k]
            prior_std[k] = cross_std[k]
        else:
            prior_mean[k] = 0.0
            prior_std[k] = cross_std[k] * 0.5

    if new_feature_indices:
        existing_stds = [
            prior_std[k]
            for k in range(total_params)
            if k not in new_feature_indices
        ]
        avg_std = np.mean(existing_stds) if existing_stds else 1.0
        for k in new_feature_indices:
            prior_std[k] = avg_std

    prior_std = np.maximum(prior_std, 1e-8)
    prior_cov = np.diag(prior_std**2)

    logger.info(
        "Prior constructed: %d params, %.4f%% significant",
        total_params,
        100.0 * np.mean(pooled_p < alpha_level),
    )

    return prior_mean, prior_cov
