from __future__ import annotations

import logging
from collections import deque

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
        self._step_count = 0
        self._done = False
        self._current_state: StateView | None = None
        self._daily_total = 0
        self._rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
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

        # Bayesian P-success burden state
        self._failure_history: deque[bool] | None = None
        self._burden_mapping: dict[int, str] | None = None
        self._success_probs: dict[str, float] = {}
        self._has_bayesian_burden = False
        self._bayesian_window_size = 0
        self._init_bayesian_burden()

    @property
    def action_history(self) -> tuple[str, ...]:
        return tuple(self._action_history)

    def _prime_action_history(self) -> None:
        self._action_history.clear()
        for _ in range(self._action_history.maxlen or 0):
            self._action_history.append("idle")

    def _apply_rolling_advances(self, action: str, state: StateView) -> StateView:  # noqa: C901
        self._action_history.append(action)
        for name, adv in self._rolling_vars:
            if adv.mechanism == "bayesian_p_success":
                continue  # Already handled by _update_burden
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

    def _init_bayesian_burden(self) -> None:
        """Set up Bayesian P-success burden if any rolling var uses it."""
        for name, adv in self._rolling_vars:
            if adv.mechanism == "bayesian_p_success":
                if self._has_bayesian_burden:
                    logger.warning(
                        "Multiple Bayesian P-success variables found; "
                        "using '%s', ignoring '%s'",
                        self._bayesian_var,
                        name,
                    )
                    continue
                self._failure_history = deque(maxlen=adv.window_size)
                self._burden_mapping = adv.mapping
                self._bayesian_window_size = adv.window_size
                self._has_bayesian_burden = True
                self._bayesian_var = name
                self._success_probs = self._precompute_success_probs()
                logger.debug(
                    "Bayesian P-success burden enabled: window=%d, mapping=%s",
                    adv.window_size,
                    adv.mapping,
                )

    def _precompute_success_probs(  # noqa: C901, PLR0912, PLR0915
        self,
    ) -> dict[str, float]:
        """Compute P(success | state, action) for all state-action pairs.

        Uses Formula 3:
        P_success_f(s, a) = Σ P_f(v|s,a) * P_f(v|s,a) / (P_f(v|s,a) + P_f(v|s,idle))

        Falls back to empty dict (P_success = 0.5) for transition models
        without per-state-action probability tables.
        """
        result: dict[str, float] = {}

        # Try table transition lookup first
        lookup = getattr(self._transition, "lookup", None)
        if not lookup:
            # Try rule_based cache next (tuple keys = (state_val, action))
            cache = getattr(self._transition, "_cache", None)
            if cache and self._is_rule_based_cache(cache):
                return self._precompute_from_cache(cache)
            return result

        # Group entries by state (without trailing action)
        state_actions: dict[
            str, dict[str, dict[str, tuple[list[str], np.ndarray]]]
        ] = {}
        for key, factor_dists in lookup.items():
            parts = key.split("|")
            action = parts[-1]
            state_key = "|".join(parts[:-1])
            if state_key not in state_actions:
                state_actions[state_key] = {}
            state_actions[state_key][action] = factor_dists

        # Stochastic factors — those without advanced configs
        stochastic = [
            n for n, c in self._config.state.variables.items() if c.advanced is None
        ]

        for state_key, action_dists in state_actions.items():
            idle_dists = action_dists.get("idle")
            if idle_dists is None:
                continue

            for action, factor_dists in action_dists.items():
                if action == "idle":
                    continue

                factor_successes: list[float] = []
                for factor in stochastic:
                    if factor not in factor_dists or factor not in idle_dists:
                        continue

                    targets_a, probs_a = factor_dists[factor]
                    targets_i, probs_i = idle_dists[factor]

                    prob_a_map = dict(zip(targets_a, probs_a, strict=False))
                    prob_i_map = dict(zip(targets_i, probs_i, strict=False))

                    all_targets = set(targets_a) | set(targets_i)

                    p_sum = self._formula3_sum(prob_a_map, prob_i_map, all_targets)
                    factor_successes.append(p_sum)

                if factor_successes:
                    combined = sum(factor_successes) / len(factor_successes)
                else:
                    combined = 0.5

                combined_key = f"{state_key}|{action}"
                result[combined_key] = combined

        return result

    @staticmethod
    def _is_rule_based_cache(cache: dict) -> bool:
        """Check if cache has tuple keys (RuleBasedTransition)."""
        if not cache:
            return False
        first_key = next(iter(cache))
        return isinstance(first_key, tuple)

    @staticmethod
    def _formula3_sum(
        action_probs: dict[str, float],
        idle_probs: dict[str, float],
        targets: set[str],
    ) -> float:
        """Compute Formula 3: Σ P_a(t) * P_a(t) / (P_a(t) + P_i(t)).

        Edge cases:
        - Both zero → contribution = 0.5
        - Action zero, idle positive → contribution = 0
        - Action positive, idle zero → contribution = 1.0 * P_a(t) = P_a(t)
        """
        p_sum = 0.0
        for tgt in targets:
            p_a = action_probs.get(tgt, 0.0)
            p_i = idle_probs.get(tgt, 0.0)

            if p_a == 0.0 and p_i == 0.0:
                p_success = 0.5
            elif p_a == 0.0:
                p_success = 0.0
            elif p_i == 0.0:
                p_success = 1.0
            else:
                p_success = p_a / (p_a + p_i)

            p_sum += p_a * p_success

        return p_sum

    def _precompute_from_cache(  # noqa: C901
        self,
        cache: dict[tuple[str, str], tuple[list[str], np.ndarray]],
    ) -> dict[str, float]:
        """Precompute P(success) from RuleBasedTransition cache."""
        result: dict[str, float] = {}
        # Rule-based cache keys are (state_val, action)
        # Group by state_val
        state_actions: dict[str, dict[str, tuple[list[str], np.ndarray]]] = {}
        for (state_val, action), (targets, probs) in cache.items():
            if state_val not in state_actions:
                state_actions[state_val] = {}
            state_actions[state_val][action] = (targets, probs)

        for state_key_val, action_dists in state_actions.items():
            idle_data = action_dists.get("idle")
            if idle_data is None:
                continue

            idle_targets, idle_probs = idle_data
            idle_map = dict(zip(idle_targets, idle_probs, strict=False))

            for action, (targets, probs) in action_dists.items():
                if action == "idle":
                    continue

                p_a_map = dict(zip(targets, probs, strict=False))
                all_tgts = set(targets) | set(idle_targets)

                p_sum = self._formula3_sum(p_a_map, idle_map, all_tgts)
                result[f"{state_key_val}|{action}"] = p_sum

        return result

    def _get_success_prob(self, state: StateView, action: str) -> float:
        """Look up the precomputed P(success) for a given (state, action)."""
        if not self._success_probs:
            return 0.5

        config_var_order = list(self._config.state.variables.keys())
        factor_values = state.factor_values
        parts = [str(factor_values.get(f, "")) for f in config_var_order]
        if getattr(self._transition, "_include_step_of_day", False):
            parts.append(str(state.step_of_day))
        key = "|".join(parts) + f"|{action}"
        return self._success_probs.get(key, 0.5)

    def _update_burden(self, state: StateView, action: str) -> StateView:
        """Perform Bayesian Bernoulli draw and update burden factor.

        Skips idle actions. Draws from Bernoulli(P_success). A failure
        (draw == 0) is appended to the failure history. Maps the current
        failure count to a burden level using the configured mapping.
        """
        if action == "idle" or self._failure_history is None:
            return state

        p_success = self._get_success_prob(state, action)
        draw = self._rng.random() < p_success
        self._failure_history.append(not draw)  # True = failure

        burden = self._get_burden_from_failures()
        return state.with_factors(**{self._bayesian_var: burden})

    def _get_burden_from_failures(self) -> str:
        """Map the current failure count to a burden level."""
        if self._failure_history is None or self._burden_mapping is None:
            return "low"
        count = sum(1 for f in self._failure_history if f)
        capped = min(count, max(self._burden_mapping.keys()))
        while capped >= 0 and capped not in self._burden_mapping:
            capped -= 1
        return self._burden_mapping[capped]

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
        if self._failure_history is not None:
            self._failure_history.clear()
            for _ in range(self._bayesian_window_size):
                self._failure_history.append(False)
        self._current_state = StateView(
            factors=dict(self._config.initial_state),
            day=0,
            step_of_day=0,
            steps_per_day=self._config.steps_per_day,
        )
        logger.debug("Environment reset: %s", self._current_state)
        return self._current_state

    def step(self, action: str) -> tuple[StateView, float, bool]:  # noqa: C901
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

        if self._has_bayesian_burden:
            state = self._update_burden(state, action)

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
