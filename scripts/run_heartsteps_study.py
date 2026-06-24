"""Run HeartSteps V2 simulation study with real NHANES data."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _get_weather(wl, meta, idx):
    if wl and meta:
        m = meta[idx]
        return wl.get_weather(m["date"], m["region"])
    return None


def _sim(
    step_part,
    g_dim,
    f_dim,
    n_days,
    n_windows,
    pi_param,
    agent_type,
    prior_mean,
    prior_cov,
    proxy_gamma,
    proxy_w,
    p_avail,
    p_sed,
    dosage_decay,
    treat_benefit,
    burden_coef,
    noise_var,
    seed,
    wctx=None,
    alpha=None,
    beta=None,
):
    from rl_health_interventions.agents.heartsteps.bayesian import BayesianRewardModel
    from rl_health_interventions.agents.heartsteps.dosage import DosageTracker
    from rl_health_interventions.agents.heartsteps.features import (
        construct_heartsteps_features,
    )
    from rl_health_interventions.agents.heartsteps.proxy import ProxyValueFunction

    rng = np.random.default_rng(seed)
    steps_flat = step_part.flatten()
    total_t = n_days * n_windows
    if alpha is None:
        alpha = np.array(
            [
                1.0,
                0.5,
                0.3,
                0.1,
                0.2,
                0.1,
                0.2,
                0.15,
                0.1,
                0.05,
                0.3,
                0.2,
                0.1,
                0.15,
                0.1,
                0.05,
                0.1,
                0.05,
            ]
        )
    if beta is None:
        beta = np.array([1.5, 2.5, 1.0, 0.5, 0.3, 0.2, 0.1, 0.15])
    model = BayesianRewardModel(
        g_dim, f_dim, prior_mean.copy(), prior_cov.copy(), noise_var
    )
    dosage = DosageTracker(dosage_decay)
    proxy = ProxyValueFunction(
        dosage_decay,
        proxy_gamma,
        p_avail,
        p_sed,
        proxy_w,
        20.0,
        0.5,
        treat_benefit,
        burden_coef,
    )
    proxy.solve()
    daily_s, total_r = 0.0, 0.0
    for t in range(total_t):
        day, window = t // n_windows, t % n_windows
        ts = window // 2
        avail = False if window in {0, 9} else rng.binomial(1, p_avail) == 1
        g, f = construct_heartsteps_features(steps_flat, t, daily_s, ts, day, wctx)
        act_int = 0
        if avail:
            if agent_type == "prop":
                bs = model.sample_beta(rng)
                a0, a1 = (
                    model.posterior_mean[:g_dim],
                    model.posterior_mean[g_dim : g_dim + f_dim],
                )
                x = dosage.value
                Q0 = float(
                    g @ a0
                    + pi_param * (f @ a1)
                    + (0 - pi_param) * (f @ bs)
                    + proxy.gamma * proxy.H(x, 0)
                )
                Q1 = float(
                    g @ a0
                    + pi_param * (f @ a1)
                    + (1 - pi_param) * (f @ bs)
                    + proxy.gamma * proxy.H(x, 1)
                )
                mq = max(Q0, Q1)
                p = float(
                    np.clip(
                        np.exp(Q1 - mq) / (np.exp(Q0 - mq) + np.exp(Q1 - mq)), 0.1, 0.2
                    )
                )
                act_int = int(rng.binomial(1, p))
            else:
                th = rng.multivariate_normal(model.posterior_mean, model.posterior_cov)
                Q0 = float(g @ th[:g_dim])
                Q1 = float(g @ th[:g_dim] + f @ th[g_dim : g_dim + f_dim])
                mq = max(Q0, Q1)
                p = float(
                    np.clip(
                        np.exp(Q1 - mq) / (np.exp(Q0 - mq) + np.exp(Q1 - mq)), 0.1, 0.2
                    )
                )
                act_int = int(rng.binomial(1, p))
        R = float(
            g @ alpha[:g_dim]
            + f @ alpha[g_dim : g_dim + f_dim]
            + act_int * f[: len(beta)] @ beta
            + rng.normal(0, np.sqrt(noise_var))
        )
        if avail:
            if agent_type == "prop":
                model.update(g, f, act_int, pi_param, R, available=True)
            else:
                phi = np.concatenate([g, pi_param * f, (act_int - pi_param) * f])
                s2i = 1.0 / noise_var
                Sinv = np.linalg.inv(model.posterior_cov)
                nc = np.linalg.inv(Sinv + s2i * np.outer(phi, phi))
                nm = nc @ (s2i * phi * R + Sinv @ model.posterior_mean)
                model._posterior_mean, model._posterior_cov = nm, nc
            dosage.update(treatment_delivered=act_int == 1)
        total_r += R
        daily_s += float(steps_flat[t])
    return total_r


def main() -> None:
    parser = argparse.ArgumentParser(description="HeartSteps V2 Simulation Study")
    parser.add_argument("--config", default="config/heartsteps.yaml")
    parser.add_argument("--data-source", choices=["real", "synthetic"], default=None)
    parser.add_argument("--n-participants", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--episode-days", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_source = args.data_source or cfg.get("data", {}).get("source", "synthetic")
    n_participants = args.n_participants or cfg.get("n_participants", 100)
    n_folds = args.n_folds or cfg.get("n_folds", 3)
    n_days = args.episode_days or cfg.get("episode_days", 42)
    n_re_runs = cfg.get("n_re_runs", 3)
    seed = args.seed or cfg.get("seed", 42)
    n_windows = cfg.get("steps_per_day", 10)

    agent_cfg = cfg["agents"][0]
    g_dim = agent_cfg.get("g_dim", 12)
    f_dim = agent_cfg.get("f_dim", 8)
    pi_param = agent_cfg.get("pi_param", 0.3)
    p_avail = agent_cfg.get("p_avail", 0.85)
    p_sed = agent_cfg.get("p_sed", 0.2)
    dosage_decay = agent_cfg.get("dosage_decay", 0.95)
    treat_benefit = agent_cfg.get("treat_benefit", 2.0)
    burden_coef = agent_cfg.get("burden_coef", 0.3)
    noise_var = agent_cfg.get("noise_variance", 1.0)

    reward_cfg = cfg.get("reward", {})
    alpha = np.array(
        reward_cfg.get(
            "alpha",
            [
                1.0,
                0.5,
                0.3,
                0.1,
                0.2,
                0.1,
                0.2,
                0.15,
                0.1,
                0.05,
                0.3,
                0.2,
                0.1,
                0.15,
                0.1,
                0.05,
                0.1,
                0.05,
                0.3,
                0.2,
            ],
        )
    )
    beta = np.array(reward_cfg.get("beta", [1.5, 2.5, 1.0, 0.5, 0.3, 0.2, 0.1, 0.15]))

    grid_cfg = cfg.get("grid", {})
    gamma_vals = grid_cfg.get("gamma", [0.0, 0.3, 0.5, 0.7, 0.9, 1.0])
    omega_vals = grid_cfg.get("omega", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    t0 = time.time()
    logger.info("HeartSteps V2 Simulation Study")
    logger.info("=" * 60)
    logger.info(
        "data=%s n_part=%d folds=%d days=%d grid=%dx%d",
        data_source,
        n_participants,
        n_folds,
        n_days,
        len(gamma_vals),
        len(omega_vals),
    )

    from rl_health_interventions.data.nhanes import NHANESLoader
    from rl_health_interventions.simulation.cross_validation import create_folds
    from rl_health_interventions.simulation.prior import construct_prior_from_steps

    data_path = cfg.get("data", {}).get(
        "data_path", "data/nhanes-step-count/csv/nhanes_1440_PAXMTSM.csv.xz"
    )
    subject_info_path = cfg.get("data", {}).get(
        "subject_info_path", "data/nhanes-step-count/subject-info.csv"
    )
    participants_csv = cfg.get("data", {}).get(
        "participants_csv", "data/nhanes/participants.csv"
    )
    loader = NHANESLoader(
        data_source=data_source,
        n_participants=n_participants,
        n_days=n_days,
        seed=seed,
        data_path=data_path,
        subject_info_path=subject_info_path,
    )
    step_data, participant_meta = loader.load()

    if os.path.exists(participants_csv):
        import pandas as pd

        pdf = pd.read_csv(participants_csv)
        meta_map = {
            int(row["seqn"]): {"date": row["date"], "region": int(row["region"])}
            for _, row in pdf.iterrows()
        }
        for m in participant_meta:
            if m["seqn"] in meta_map:
                m["date"] = meta_map[m["seqn"]]["date"]
                m["region"] = meta_map[m["seqn"]]["region"]
        logger.info("Merged participant metadata from %s", participants_csv)

    logger.info("Loaded data: %s", step_data.shape)

    weather_loader = None
    weather_path = cfg.get("data", {}).get("weather_path")
    if weather_path and data_source == "real":
        try:
            from rl_health_interventions.data.weather import WeatherLoader

            weather_loader = WeatherLoader(weather_path)
        except Exception:
            logger.warning("Could not load weather data")

    folds = create_folds(step_data.shape[0], n_folds, rng=np.random.default_rng(seed))
    all_improvements: list[float] = []
    all_results: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        logger.info(
            "Fold %d (train=%d, test=%d)", fold_idx + 1, len(train_idx), len(test_idx)
        )
        prior_mean, prior_cov = construct_prior_from_steps(
            train_idx.tolist(),
            step_data,
            g_dim=g_dim,
            f_dim=f_dim,
            n_days=n_days,
            n_windows=n_windows,
            pi_param=pi_param,
            p_avail=p_avail,
            weather_loader=weather_loader,
            participant_meta=participant_meta,
            rng=np.random.default_rng(
                int(np.random.default_rng(seed).integers(0, 2**31))
            ),
        )
        logger.info(
            "  Prior mean norm: %.4f, cov trace: %.4f",
            np.linalg.norm(prior_mean),
            np.trace(prior_cov),
        )

        best_g, best_w, best_r = 0.0, 0.0, -float("inf")
        for gv in gamma_vals:
            for wv in omega_vals:
                rews = []
                for p in train_idx[:5]:
                    for _ in range(2):
                        wctx = _get_weather(weather_loader, participant_meta, int(p))
                        rews.append(
                            _sim(
                                step_data[int(p)],
                                g_dim,
                                f_dim,
                                n_days,
                                n_windows,
                                pi_param,
                                "prop",
                                prior_mean,
                                prior_cov,
                                gv,
                                wv,
                                p_avail,
                                p_sed,
                                dosage_decay,
                                treat_benefit,
                                burden_coef,
                                noise_var,
                                int(np.random.default_rng(seed).integers(0, 2**31)),
                                wctx,
                                alpha,
                                beta,
                            )
                        )
                mr = np.mean(rews) if rews else 0
                if mr > best_r:
                    best_r, best_g, best_w = mr, gv, wv

        logger.info("  Best gamma=%.2f w=%.2f", best_g, best_w)
        fold_imps: list[float] = []
        for p_idx in test_idx:
            prop_r, band_r = [], []
            for _ in range(n_re_runs):
                wctx = _get_weather(weather_loader, participant_meta, int(p_idx))
                seed_i = int(np.random.default_rng(seed).integers(0, 2**31))
                prop_r.append(
                    _sim(
                        step_data[int(p_idx)],
                        g_dim,
                        f_dim,
                        n_days,
                        n_windows,
                        pi_param,
                        "prop",
                        prior_mean,
                        prior_cov,
                        best_g,
                        best_w,
                        p_avail,
                        p_sed,
                        dosage_decay,
                        treat_benefit,
                        burden_coef,
                        noise_var,
                        seed_i,
                        wctx,
                        alpha,
                        beta,
                    )
                )
                seed_i = int(np.random.default_rng(seed).integers(0, 2**31))
                band_r.append(
                    _sim(
                        step_data[int(p_idx)],
                        g_dim,
                        f_dim,
                        n_days,
                        n_windows,
                        pi_param,
                        "bandit",
                        prior_mean,
                        prior_cov,
                        0.0,
                        0.0,
                        p_avail,
                        p_sed,
                        dosage_decay,
                        treat_benefit,
                        burden_coef,
                        noise_var,
                        seed_i,
                        wctx,
                        alpha,
                        beta,
                    )
                )
            imp = float(np.mean(prop_r) - np.mean(band_r))
            fold_imps.append(imp)
            all_improvements.append(imp)
            all_results.append(
                {
                    "fold": fold_idx,
                    "participant": int(p_idx),
                    "proposed": float(np.mean(prop_r)),
                    "bandit": float(np.mean(band_r)),
                    "improvement": imp,
                }
            )
            logger.info(
                "  P%02d: prop=%.1f bandit=%.1f imp=%+.1f step_mean=%.0f",
                int(p_idx),
                np.mean(prop_r),
                np.mean(band_r),
                imp,
                step_data[int(p_idx)].mean(),
            )

    arr = np.array(all_improvements)
    n_imp = int(np.sum(arr > 0))
    logger.info("=" * 60)
    logger.info(
        "RESULTS: %d/%d improved (%.1f%%)", n_imp, len(arr), 100 * n_imp / len(arr)
    )
    logger.info(
        "Mean: %+.2f  Median: %+.2f", float(np.mean(arr)), float(np.median(arr))
    )
    logger.info(
        "Range: [%.2f, %.2f]  Time: %.1fs",
        float(np.min(arr)),
        float(np.max(arr)),
        time.time() - t0,
    )

    with open("results_heartsteps.json", "w") as f:
        json.dump(
            {
                "config": {
                    "data_source": data_source,
                    "n_participants": n_participants,
                    "n_folds": n_folds,
                    "episode_days": n_days,
                    "g_dim": g_dim,
                    "f_dim": f_dim,
                },
                "summary": {
                    "n_improved": n_imp,
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                },
                "results": all_results,
            },
            f,
            indent=2,
        )
    logger.info("Results saved to results_heartsteps.json")


if __name__ == "__main__":
    main()
