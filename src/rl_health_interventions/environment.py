from __future__ import annotations

import logging
from collections import deque
from typing import cast

import numpy as np

from rl_health_interventions import make_reward, make_transition
from rl_health_interventions.config.schemas import (
    CyclicAdvance,
    MDPConfig,
    RollingWindowCountAdvance,
)
from rl_health_interventions.state import StateView

logger = logging.getLogger(__name__)

_MIDPOINT: dict[str, int] = {
    "inactive": 400,
    "moderate": 1200,
    "active": 2000,
}

_DAILY_INACTIVE_UPPER = 4000
_DAILY_MODERATE_UPPER = 8000


def _bin_daily(total: int) -> str:
    if total < _DAILY_INACTIVE_UPPER:
        return "inactive"
    if total <= _DAILY_MODERATE_UPPER:
        return "moderate"
    return "active"


class Environment:
    def __init__(self, config: MDPConfig, seed: int = 42) -> None:
        self._config = config
        self._transition = make_transition(config, seed=seed)
        self._reward = make_reward(config)
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._done = False
        self._current_state: StateView | None = None
        self._daily_total = 0
        self._cyclic_vars: list[tuple[str, CyclicAdvance]] = [
            (n, c.advanced)
            for n, c in config.state.variables.items()
            if isinstance(c.advanced, CyclicAdvance)
        ]
        self._rolling_vars: list[tuple[str, RollingWindowCountAdvance]] = [
            (n, c.advanced)
            for n, c in config.state.variables.items()
            if isinstance(c.advanced, RollingWindowCountAdvance)
        ]
        window_sizes = [adv.window_size for _, adv in self._rolling_vars]
        max_window = max(window_sizes) if window_sizes else 3
        self._action_history: deque[tuple[str, str]] = deque(maxlen=max_window)
        self._prime_action_history()
        self._p_success: dict[str, float] = {}
        self._precompute_p_success()

    @property
    def action_history(self) -> tuple[str, ...]:
        return tuple(action for _, action in self._action_history)

    def _prime_action_history(self) -> None:
        self._action_history.clear()
        for _ in range(self._action_history.maxlen or 0):
            self._action_history.append(("", "idle"))

    def _precompute_p_success(self) -> None:  # noqa: C901, PLR0911, PLR0912, PLR0915
        """Precompute P-success for each (state_key, action) pair.

        Flat format: P(success | s, a) = 1 - Σ P(ns | s, a) * P(ns | s, idle)

        Per-factor format: combines per-factor overlaps across all stochastic
        factors so no single factor's distribution is privileged.

        Requires the transition model to expose ``within_day`` tables
        (``TableTransition``).
        """
        if not hasattr(self._transition, "within_day"):
            return
        tm = self._transition
        actions = list(self._config.actions.keys())
        if "idle" not in actions:
            return

        stochastic = [
            n for n, c in self._config.state.variables.items() if c.advanced is None
        ]
        factor_value_lists = []
        for name in stochastic:
            var_cfg = self._config.state.variables.get(name)
            if var_cfg is None:
                continue
            factor_value_lists.append((name, var_cfg.names))

        from itertools import product  # noqa: PLC0415

        # Per-factor format: use _pf_wd directly, combine across all factors
        if hasattr(tm, "_per_factor") and tm._per_factor:
            pf_wd = getattr(tm, "_pf_wd", None)
            if not pf_wd:
                return
            wd_first = pf_wd[0]
            for combo in product(*(values for _, values in factor_value_lists)):
                state_key = "|".join(combo)
                for action in actions:
                    if action == "idle":
                        continue
                    factor_overlaps = []
                    for i, name in enumerate(stochastic):
                        if name not in wd_first:
                            continue
                        fv = combo[i]
                        idle_key = f"{fv}|idle"
                        action_key = f"{fv}|{action}"
                        if (
                            idle_key not in wd_first[name]
                            or action_key not in wd_first[name]
                        ):
                            continue
                        idle_targets, idle_probs = wd_first[name][idle_key]
                        action_targets, action_probs = wd_first[name][action_key]
                        all_targets = sorted(set(idle_targets) | set(action_targets))
                        idle_map = dict(zip(idle_targets, idle_probs, strict=True))
                        action_map = dict(
                            zip(action_targets, action_probs, strict=True)
                        )
                        p_idle = np.array([idle_map.get(t, 0.0) for t in all_targets])
                        p_action = np.array(
                            [action_map.get(t, 0.0) for t in all_targets]
                        )
                        overlap = float(np.sum(p_action * p_idle))
                        factor_overlaps.append(overlap)
                    if factor_overlaps:
                        p_success = 1.0 - float(np.prod(factor_overlaps))
                        key = f"{state_key}|{action}"
                        self._p_success[key] = max(0.0, min(1.0, p_success))
            return

        # Flat format (sprint1): single combined distribution per key
        within_day = cast("list", tm.within_day)
        if not within_day:
            return
        wd_first: dict[str, tuple] = within_day[0]

        for combo in product(*(values for _, values in factor_value_lists)):
            state_key = "|".join(combo)

            idle_key = state_key + "|idle"
            if idle_key not in wd_first:
                continue
            idle_targets, idle_probs = wd_first[idle_key]

            for action in actions:
                if action == "idle":
                    continue
                action_key = state_key + "|" + action
                if action_key not in wd_first:
                    continue
                action_targets, action_probs = wd_first[action_key]

                # Align probability vectors over same target set
                all_targets = sorted(set(idle_targets) | set(action_targets))
                idle_map = dict(zip(idle_targets, idle_probs, strict=True))
                action_map = dict(zip(action_targets, action_probs, strict=True))

                p_idle = np.array([idle_map.get(t, 0.0) for t in all_targets])
                p_action = np.array([action_map.get(t, 0.0) for t in all_targets])

                # P(success) = 1 - Σ_t P(t|s,a) * P(t|s,idle)
                p_success = 1.0 - float(np.sum(p_action * p_idle))
                self._p_success[f"{state_key}|{action}"] = max(0.0, min(1.0, p_success))

    def _apply_rolling_advances(self, action: str, state: StateView) -> StateView:  # noqa: C901, PLR0912
        state_key = "|".join(
            getattr(state, n)
            for n, c in self._config.state.variables.items()
            if c.advanced is None
        )
        self._action_history.append((state_key, action))
        for name, adv in self._rolling_vars:
            window = list(self._action_history)[-adv.window_size :]
            count = 0
            for cond in adv.conditions:
                if cond.factor == "action":
                    if self._p_success and name == "burden":
                        # P-success burden: idle never counts as failure
                        for hk_state_key, a in window:
                            if a == "idle":
                                continue
                            p_key = f"{hk_state_key}|{a}"
                            p_success = self._p_success.get(p_key, 0.5)
                            if self._rng.random() >= p_success:
                                count += 1
                    else:
                        count += sum(1 for _, a in window if a in cond.values)
                else:
                    fv = getattr(state, cond.factor, None)
                    if fv in cond.values:
                        count += len(window)
            capped = min(count, max(adv.mapping.keys()))
            state = state.with_factors(**{name: adv.mapping[capped]})
        return state

    def _apply_cyclic_advances(self, state: StateView) -> StateView:
        for name, adv in self._cyclic_vars:
            val = adv.pattern[state.day % len(adv.pattern)]
            state = state.with_factors(**{name: val})
        return state

    def reset(self) -> StateView:
        self._step_count = 0
        self._done = False
        self._daily_total = 0
        self._prime_action_history()
        self._current_state = StateView(
            factors=dict(self._config.initial_state),
            day=0,
            step_of_day=0,
            steps_per_day=self._config.steps_per_day,
        )
        logger.debug("Environment reset: %s", self._current_state)
        return self._current_state

    def step(self, action: str) -> tuple[StateView, float, bool]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._current_state is None:
            raise RuntimeError("Call reset() before step().")

        step_idx = self._step_count % self._config.steps_per_day
        state = self._current_state

        if step_idx == 0:
            state = state.with_factors(step_bin_daily=_bin_daily(self._daily_total))
            self._daily_total = 0

        updates = self._transition.transition(state, action)
        state = state.with_factors(**updates)

        if hasattr(state, "step_bin"):
            self._daily_total += _MIDPOINT.get(state.step_bin, 0)

        state = self._apply_cyclic_advances(state)
        state = self._apply_rolling_advances(action, state)

        self._current_state = state
        reward, _ = self._reward.reward(self._current_state, action, step_idx)
        self._step_count += 1
        self._current_state = self._current_state.with_advance()
        self._done = (
            self._step_count >= self._config.steps_per_day * self._config.episode_days
        )

        logger.debug(
            "Step %d: action=%s, next=%s, reward=%.2f, done=%s",
            self._step_count,
            action,
            self._current_state,
            reward,
            self._done,
        )
        return self._current_state, reward, self._done
