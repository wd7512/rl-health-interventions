from __future__ import annotations

import logging
from collections import deque

from rl_health_interventions import make_reward, make_transition
from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView

logger = logging.getLogger(__name__)

_MID_POINTS = {"inactive": 400.0, "moderate": 1200.0, "active": 2000.0}
_DAILY_THRESHOLDS = [(4000.0, "inactive"), (8000.0, "moderate")]


def _mid_point(step_bin: str) -> float:
    return _MID_POINTS.get(step_bin, 0.0)


def _bin_daily(total: float) -> str:
    for threshold, name in _DAILY_THRESHOLDS:
        if total < threshold:
            return name
    return "active"


class Environment:
    def __init__(self, config: MDPConfig, seed: int = 42) -> None:
        self._config = config
        self._transition = make_transition(config, seed=seed)
        self._reward = make_reward(config)
        self._step_count = 0
        self._done = False
        self._current_state: StateView | None = None
        self._action_history: deque[str] = deque(maxlen=3)
        self._factor_names: set[str] = set(config.factor_configs.keys())
        self._daily_total: float = 0.0

    def reset(self) -> StateView:
        self._step_count = 0
        self._done = False
        self._action_history.clear()
        self._daily_total = 0.0
        self._current_state = StateView(
            dict(self._config.initial_state),
            day=0,
            step_of_day=0,
            steps_per_day=self._config.steps_per_day,
        )
        logger.debug("Environment reset: %s", self._current_state)
        return self._current_state

    def _compute_burden(self) -> str:
        """Compute burden from last 3 actions."""
        non_idle = sum(1 for a in self._action_history if a != "idle")
        if non_idle == 0:
            return "low"
        elif non_idle == 1:
            return "medium"
        else:
            return "high"

    def step(self, action: str) -> tuple[StateView, float, bool]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._current_state is None:
            raise RuntimeError("Call reset() before step().")

        step_of_day = self._step_count % self._config.steps_per_day

        # 1a. At day boundary, compute daily bin and reset accumulator
        daily_bin = None
        if step_of_day == 0:
            daily_bin = _bin_daily(self._daily_total)
            self._daily_total = 0.0

        # 1b. Transition returns factor updates
        factor_updates = self._transition._transition_updates(
            self._current_state, action, daily_bin=daily_bin
        )

        # 1c. Accumulate daily step total
        if "step_bin" in factor_updates:
            self._daily_total += _mid_point(factor_updates["step_bin"])

        # 2. Inject day_of_week if configured (Sprint1+)
        if "day_of_week" in self._factor_names:
            days_elapsed = (self._step_count + 1) // self._config.steps_per_day
            if days_elapsed >= 5 and (days_elapsed % 7) in (5, 6):
                factor_updates["day_of_week"] = "weekend"
            else:
                factor_updates["day_of_week"] = "weekday"

        # 3. Update action history and inject burden if configured (Sprint1+)
        self._action_history.append(action)
        if "burden" in self._factor_names:
            factor_updates["burden"] = self._compute_burden()

        # 4. Advance counters
        self._step_count += 1
        new_step_of_day = self._step_count % self._config.steps_per_day
        new_day = self._step_count // self._config.steps_per_day

        # 5. Build single StateView with all updates + advance
        next_factors = dict(self._current_state.factor_values)
        next_factors.update(factor_updates)
        next_state = StateView(
            next_factors,
            day=new_day,
            step_of_day=new_step_of_day,
            steps_per_day=self._config.steps_per_day,
        )

        # 6. Compute reward
        reward, _ = self._reward.reward(next_state, action, step_of_day)

        if self._step_count >= self._config.steps_per_day * self._config.episode_days:
            self._done = True

        self._current_state = next_state

        logger.debug(
            "Step %d: action=%s, next=%s, reward=%.2f, done=%s",
            self._step_count,
            action,
            self._current_state,
            reward,
            self._done,
        )
        return self._current_state, reward, self._done
