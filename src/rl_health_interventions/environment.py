from __future__ import annotations

import logging
from collections import deque

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
        self._action_history: deque[str] = deque(maxlen=max_window)
        self._prime_action_history()

    def _prime_action_history(self) -> None:
        self._action_history.clear()
        for _ in range(self._action_history.maxlen or 0):
            self._action_history.append("idle")

    def _apply_rolling_advances(self, action: str, state: StateView) -> StateView:
        self._action_history.append(action)
        for name, adv in self._rolling_vars:
            window = list(self._action_history)[-adv.window_size :]
            count = 0
            for cond in adv.conditions:
                if cond.factor == "action":
                    count += sum(1 for a in window if a in cond.values)
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
