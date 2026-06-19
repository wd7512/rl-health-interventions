from __future__ import annotations

import dataclasses
import logging

from rl_health_interventions import make_reward, make_transition
from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView

logger = logging.getLogger(__name__)


class Environment:
    def __init__(self, config: MDPConfig, seed: int = 42) -> None:
        self._config = config
        self._transition = make_transition(config, seed=seed)
        self._reward = make_reward(config)
        self._step_count = 0
        self._done = False
        self._current_state: StateView | None = None

    def reset(self) -> StateView:
        self._step_count = 0
        self._done = False
        self._current_state = StateView(
            activity=self._config.initial_state,
            day=0,
            step_of_day=0,
            steps_per_day=self._config.steps_per_day,
            steps=self._config.initial_steps,
            weight=self._config.initial_weight,
            time_of_day=0,
            day_of_week=0,
        )
        logger.debug("Environment reset: %s", self._current_state)
        return self._current_state

    def step(self, action: str) -> tuple[StateView, float, bool]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._current_state is None:
            raise RuntimeError("Call reset() before step().")

        next_state = self._transition.transition(self._current_state, action)

        step_idx = self._step_count % self._config.steps_per_day
        reward, _ = self._reward.reward(next_state, action, step_idx)

        self._step_count += 1

        # Overwrite deterministic fields
        new_step_of_day = self._step_count % self._config.steps_per_day
        new_day = self._step_count // self._config.steps_per_day
        self._current_state = dataclasses.replace(
            next_state,
            day=new_day,
            step_of_day=new_step_of_day,
            time_of_day=int(new_step_of_day * 24 / self._config.steps_per_day),
            day_of_week=new_day % 7,
        )

        if self._step_count >= self._config.steps_per_day * self._config.episode_days:
            self._done = True

        logger.debug(
            "Step %d: action=%s, next=%s, reward=%.2f, done=%s",
            self._step_count,
            action,
            self._current_state,
            reward,
            self._done,
        )
        return self._current_state, reward, self._done
