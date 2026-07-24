from __future__ import annotations

import itertools
import logging

import numpy as np

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.transitions.table_transition import TableTransition

logger = logging.getLogger(__name__)

_WEEKDAY_PATTERN = [
    "weekday",
    "weekday",
    "weekday",
    "weekday",
    "weekday",
    "weekend",
    "weekend",
]
_BURDEN_MAP = {0: "low", 1: "medium", 2: "high", 3: "high"}


def _count_non_idle(hist: tuple[str, ...]) -> int:
    return sum(1 for a in hist if a != "idle")


def _action_penalty(action: str) -> float:
    return 0.05 if action != "idle" else 0.0


class OptimalBound:
    def __init__(
        self, config_path: str, alpha: float | None = None, table_dir: str | None = None
    ) -> None:
        self._config = load_config(config_path)
        if table_dir is not None:
            self._config.transition_model.table_dir = table_dir
        self._alpha = (
            alpha
            if alpha is not None
            else self._config.reward.constants.get("alpha", 0.9)
        )
        self._tm = TableTransition(self._config, seed=42)
        self._sb_names = self._config.state.variables["step_bin"].names
        self._sleep_names = self._config.state.variables["sleep"].names
        self._action_names = list(self._config.actions.keys())
        self._sod = self._config.steps_per_day
        self._T = self._config.episode_days * self._sod
        self._histories = list(itertools.product(self._action_names, repeat=3))
        self._hist_idx = {h: i for i, h in enumerate(self._histories)}
        self._nsb = len(self._sb_names)
        self._nsl = len(self._sleep_names)
        self._nh = len(self._histories)
        self._na = len(self._action_names)
        self._ns = self._nsb * self._nsl * self._nh

        # TODO: full Q/pi tables over all timesteps won't scale to 9000d horizons.
        # Consider rolling V with periodic policy for memory-bounded backward induction.
        self._states: list[tuple[str, str, tuple[str, ...]]] = []
        self._state_idx: dict[tuple[str, str, tuple[str, ...]], int] = {}
        idx = 0
        for sb in self._sb_names:
            for sl in self._sleep_names:
                for hist in self._histories:
                    self._states.append((sb, sl, hist))
                    self._state_idx[(sb, sl, hist)] = idx
                    idx += 1

        self._V = np.zeros((self._T + 1, self._ns))
        self._Q = np.zeros((self._T, self._ns, self._na))
        self._pi = np.zeros((self._T, self._ns), dtype=np.int64)

    def _sb_val(self, sb: str) -> float:
        return self._config.reward.variables["step_bin_value"].mapping[sb]

    def _sl_val(self, sl: str) -> float:
        return self._config.reward.variables["sleep_value"].mapping[sl]

    def _burden(self, hist: tuple[str, ...]) -> str:
        return _BURDEN_MAP[min(_count_non_idle(hist), 3)]

    def run(self) -> None:
        sod = self._sod
        sb_names = self._sb_names
        sl_names = self._sleep_names
        a_names = self._action_names
        db = self._tm.day_boundary
        wd = self._tm.within_day
        nh = self._nh
        nsl = self._nsl

        for t in range(self._T - 1, -1, -1):
            step = t % sod
            dow = _WEEKDAY_PATTERN[(t // sod) % 7]
            for si in range(self._ns):
                sbi = si // (nsl * nh)
                sli = (si // nh) % nsl
                hi = si % nh
                sb = sb_names[sbi]
                sl = sl_names[sli]
                hist = self._histories[hi]
                b = self._burden(hist)

                for ai in range(self._na):
                    a = a_names[ai]
                    nh_t = (hist[1], hist[2], a)
                    nhi = self._hist_idx[nh_t]

                    if step == 0:
                        dk = "|".join([sb, b, dow, sl])
                        if dk not in db:
                            self._Q[t, si, ai] = -1e9
                            continue
                        sl_tgts, sl_pr = db[dk]
                        wt = wd[step]
                        q = 0.0
                        for sl_t, sl_p in zip(sl_tgts, sl_pr):
                            sl_t = str(sl_t)
                            wk = "|".join([sb, b, a, dow, sl_t])
                            if wk not in wt:
                                q = -1e9
                                break
                            sb_tgts, sb_pr = wt[wk]
                            zsli = sl_names.index(sl_t)
                            r_sl = self._sl_val(sl_t)
                            for sb_t, sb_p in zip(sb_tgts, sb_pr):
                                sb_t = str(sb_t)
                                zsbi = sb_names.index(sb_t)
                                r_sb = self._sb_val(sb_t)
                                r = (
                                    self._alpha * r_sb
                                    + (1.0 - self._alpha) * r_sl
                                    - _action_penalty(a)
                                )
                                zs = zsbi * nsl * nh + zsli * nh + nhi
                                q += sl_p * sb_p * (r + self._V[t + 1, zs])
                        self._Q[t, si, ai] = q
                    else:
                        wt = wd[step]
                        wk = "|".join([sb, b, a, dow, sl])
                        if wk not in wt:
                            self._Q[t, si, ai] = -1e9
                            continue
                        sb_tgts, sb_pr = wt[wk]
                        zsli = sli
                        q = 0.0
                        for sb_t, sb_p in zip(sb_tgts, sb_pr):
                            sb_t = str(sb_t)
                            zsbi = sb_names.index(sb_t)
                            r_sb = self._sb_val(sb_t)
                            r_sl = self._sl_val(sl)
                            r = (
                                self._alpha * r_sb
                                + (1.0 - self._alpha) * r_sl
                                - _action_penalty(a)
                            )
                            zs = zsbi * nsl * nh + zsli * nh + nhi
                            q += sb_p * (r + self._V[t + 1, zs])
                        self._Q[t, si, ai] = q

                ba = int(np.argmax(self._Q[t, si]))
                self._V[t, si] = self._Q[t, si, ba]
                self._pi[t, si] = ba

    def optimal_value(self) -> float:
        init = self._config.initial_state
        return float(
            self._V[
                0,
                self._state_idx[
                    (init["step_bin"], init["sleep"], ("idle", "idle", "idle"))
                ],
            ]
        )

    def report(self) -> dict:
        v0 = self.optimal_value()
        return {
            "optimal_value": v0,
            "per_step": v0 / self._T,
            "n_states": self._ns,
            "n_timesteps": self._T,
        }

    def greedy_oracle(self, seeds: int = 10) -> dict:
        from rl_health_interventions.environment import Environment

        db = self._tm.day_boundary
        wd = self._tm.within_day
        totals = []
        for s in range(seeds):
            env = Environment(self._config, seed=s)
            st = env.reset()
            ep = 0.0
            for t in range(self._T):
                step = t % self._sod
                sb = str(st.step_bin)
                sl = str(st.sleep)
                dow = _WEEKDAY_PATTERN[(t // self._sod) % 7]
                hist = tuple(env.action_history)
                b = self._burden(hist)
                best_a = self._action_names[0]
                best_er = -1e9
                for a in self._action_names:
                    if step == 0:
                        dk = "|".join([sb, b, dow, sl])
                        if dk in db:
                            sl_tgts, sl_pr = db[dk]
                            esb = 0.0
                            esl = 0.0
                            for i_sl, sl_t in enumerate(sl_tgts):
                                esl += sl_pr[i_sl] * self._sl_val(str(sl_t))
                                wk = "|".join([sb, b, a, dow, str(sl_t)])
                                if wk in wd[step]:
                                    stg, spr = wd[step][wk]
                                    for j, sbt in enumerate(stg):
                                        esb += (
                                            sl_pr[i_sl]
                                            * spr[j]
                                            * self._sb_val(str(sbt))
                                        )
                            er = (
                                self._alpha * esb
                                + (1.0 - self._alpha) * esl
                                - _action_penalty(a)
                            )
                        else:
                            er = -1e9
                    else:
                        wk = "|".join([sb, b, a, dow, sl])
                        if wk in wd[step]:
                            stg, spr = wd[step][wk]
                            esb = sum(
                                spr[j] * self._sb_val(str(sbt))
                                for j, sbt in enumerate(stg)
                            )
                            er = (
                                self._alpha * esb
                                + (1.0 - self._alpha) * self._sl_val(sl)
                                - _action_penalty(a)
                            )
                        else:
                            er = -1e9
                    if er > best_er:
                        best_er = er
                        best_a = a
                ns, rw, done = env.step(best_a)
                ep += rw
                st = ns
            totals.append(ep)
        a = np.array(totals)
        return {"mean": float(a.mean()), "std": float(a.std()), "n_seeds": seeds}

    def policy_activity(self, seed: int = 42) -> dict:
        from rl_health_interventions.environment import Environment

        env = Environment(self._config, seed=seed)
        st = env.reset()
        actions: list[str] = []
        for t in range(self._T):
            sb = str(st.step_bin)
            sl = str(st.sleep)
            hist = tuple(env.action_history)
            dk = (sb, sl, hist)
            if dk not in self._state_idx:
                break
            a = self._action_names[self._pi[t, self._state_idx[dk]]]
            actions.append(a)
            ns, rw, done = env.step(a)
            st = ns
        total = len(actions)
        counts: dict[str, int] = {}
        for a in actions:
            counts[a] = counts.get(a, 0) + 1
        idle_pct = 100.0 * counts.get("idle", 0) / total if total > 0 else 0.0
        return {
            "activity_pct": 100.0 - idle_pct,
            "idle_pct": idle_pct,
            "action_distribution": {
                a: 100.0 * c / total for a, c in sorted(counts.items())
            },
            "n_steps": total,
        }

    def simulate_trajectories(self, seeds: int = 10) -> dict:
        from rl_health_interventions.environment import Environment

        totals = []
        for s in range(seeds):
            env = Environment(self._config, seed=s)
            st = env.reset()
            ep = 0.0
            for t in range(self._T):
                sb = str(st.step_bin)
                sl = str(st.sleep)
                hist = tuple(env.action_history)
                dk = (sb, sl, hist)
                if dk not in self._state_idx:
                    break
                a = self._action_names[self._pi[t, self._state_idx[dk]]]
                ns, rw, done = env.step(a)
                ep += rw
                st = ns
            totals.append(ep)
        a = np.array(totals)
        return {"mean": float(a.mean()), "std": float(a.std()), "n_seeds": seeds}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sprint1_bootstrap_extensions.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    bound = OptimalBound(
        f"docs/experimental_phases/sprint1_bootstrap/configs/{args.config}"
    )
    logger.info(
        "Computing optimal DP bound: %d states x %d timesteps...", bound._ns, bound._T
    )
    bound.run()

    r = bound.report()
    g = bound.greedy_oracle(seeds=10)
    t = bound.simulate_trajectories(seeds=10)

    logger.info("\n" + "=" * 60)
    logger.info("Optimal DP Bound (backward induction)")
    logger.info("=" * 60)
    logger.info("States:          %d", r["n_states"])
    logger.info("Timesteps:       %d", r["n_timesteps"])
    logger.info("Optimal value:   %.2f (%.4f/step)", r["optimal_value"], r["per_step"])
    logger.info(
        "Greedy oracle:   %.2f +- %.2f (%.4f/step)",
        g["mean"],
        g["std"],
        g["mean"] / bound._T,
    )
    logger.info(
        "DP trajectories: %.2f +- %.2f (%.4f/step)",
        t["mean"],
        t["std"],
        t["mean"] / bound._T,
    )
    logger.info("Idle bound:      380.40 (0.8453/step)")
    logger.info("Std UCB (90d):   358.10 (0.7958/step)")

    # Optimal policy summary
    init = bound._config.initial_state
    init_idx = bound._state_idx[
        (init["step_bin"], init["sleep"], ("idle", "idle", "idle"))
    ]
    logger.info("\n" + "-" * 60)
    logger.info("Optimal actions at initial state (first 10 steps)")
    logger.info("%-6s %-6s %-6s  %-24s %-10s", "t", "sod", "day", "action", "Q-val")
    logger.info("-" * 60)
    for tt in range(min(10, bound._T)):
        a = bound._action_names[bound._pi[tt, init_idx]]
        logger.info(
            "%-6d %-6d %-6d  %-24s %-10.4f",
            tt,
            tt % bound._sod,
            tt // bound._sod,
            a,
            bound._Q[tt, init_idx, bound._pi[tt, init_idx]],
        )

    # Count how often each action is chosen across the optimal policy
    action_counts = dict.fromkeys(bound._action_names, 0)
    for tt in range(bound._T):
        action_counts[bound._action_names[bound._pi[tt, init_idx]]] += 1
    logger.info("\n" + "-" * 60)
    logger.info("Optimal action distribution (initial state, all T=%d steps)", bound._T)
    total = sum(action_counts.values())
    for a, c in sorted(action_counts.items(), key=lambda x: -x[1]):
        logger.info("  %-24s %5d (%.1f%%)", a, c, 100 * c / total)

    logger.info(
        "\nKEY: %.2f | %.2f +- %.2f | %.2f | %.2f",
        r["optimal_value"],
        t["mean"],
        t["std"],
        g["mean"],
        380.40,
    )


if __name__ == "__main__":
    main()
