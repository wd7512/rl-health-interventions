from __future__ import annotations

import json
import logging
from itertools import product
from pathlib import Path

import numpy as np
from typing_extensions import override

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView
from rl_health_interventions.transitions._base import TransitionModel

logger = logging.getLogger(__name__)


class RandomTransitionSA(TransitionModel):
    """Random transition with per-(factor_value, action) Dirichlet tables.

    Unlike ``RandomTransition`` which uses a single Dirichlet draw per factor
    (same probabilities for all state-action pairs), this model draws a
    **separate** Dirichlet sample for each (state_value, action) pair.  This
    enables Bayesian P-success computation: the probability of a non-idle
    action "succeeding" depends on how its transition distribution differs from
    the idle action's distribution for the same state.
    """

    def __init__(self, config: MDPConfig, seed: int = 42) -> None:
        super().__init__(config, seed=seed)
        self._rng = np.random.default_rng(seed)
        # Per-factor day_boundary: {factor_name: {factor_value: (targets, probs)}}
        self._day_boundary: dict[str, dict[str, tuple[list[str], np.ndarray]]] = {}
        # Per-factor within_day tables
        # list[{factor_name: {"{fv}|{action}": (targets, probs)}}]
        self._within_day: list[
            dict[str, dict[str, tuple[list[str], np.ndarray]]]
        ] = []
        self._build_tables()

    @property
    def day_boundary(self) -> dict[str, tuple[list[str], np.ndarray]]:
        """Flat day_boundary table for BootstrapTransition compatibility.

        Keys are ``{fv1}|{fv2}|...`` (stochastic factors only).
        Each entry's distribution is the first factor's at that combo.
        """
        result: dict[str, tuple[list[str], np.ndarray]] = {}
        stochastic = self._stochastic_factors
        factor_value_lists = []
        for name in stochastic:
            var_cfg = self._config.state.variables.get(name)
            if var_cfg is None:
                continue
            factor_value_lists.append(var_cfg.names)

        for combo in product(*factor_value_lists):
            key = "|".join(combo)
            first_name = stochastic[0]
            first_val = combo[0]
            if (
                first_name in self._day_boundary
                and first_val in self._day_boundary[first_name]
            ):
                result[key] = self._day_boundary[first_name][first_val]
        return result

    @property
    def within_day(self) -> list[dict[str, tuple[list[str], np.ndarray]]]:  # noqa: C901, PLR0912
        """Flat within_day tables for BootstrapTransition compatibility.

        Keys are ``{fv1}|{fv2}|...|{action}``.
        """
        result_list: list[dict[str, tuple[list[str], np.ndarray]]] = []
        stochastic = self._stochastic_factors
        actions = list(self._config.actions.keys())
        factor_value_lists = []
        for name in stochastic:
            var_cfg = self._config.state.variables.get(name)
            if var_cfg is None:
                continue
            factor_value_lists.append(var_cfg.names)

        for step_tables in self._within_day:
            flat: dict[str, tuple[list[str], np.ndarray]] = {}
            for combo in product(*factor_value_lists):
                for action in actions:
                    full_key = "|".join(combo) + "|" + action
                    for name, fv in zip(stochastic, combo, strict=True):
                        sa_key = f"{fv}|{action}"
                        if name in step_tables and sa_key in step_tables[name]:
                            flat[full_key] = step_tables[name][sa_key]
                            break
            result_list.append(flat)
        return result_list

    def _build_tables(self) -> None:  # noqa: C901, PLR0912
        stochastic = self._stochastic_factors
        actions = list(self._config.actions.keys())

        # day_boundary: one Dirichlet draw per (factor, factor_value)
        for name in stochastic:
            var_cfg = self._config.state.variables.get(name)
            if var_cfg is None:
                continue
            self._day_boundary[name] = {}
            for fv in var_cfg.names:
                n_values = len(var_cfg.names)
                probs = self._rng.dirichlet([1.0] * n_values)
                self._day_boundary[name][fv] = (var_cfg.names, probs)

        # within_day: one Dirichlet draw per (factor, factor_value, action)
        for _step_idx in range(self._config.steps_per_day):
            step_tables: dict[str, dict[str, tuple[list[str], np.ndarray]]] = {}
            for name in stochastic:
                var_cfg = self._config.state.variables.get(name)
                if var_cfg is None:
                    continue
                step_tables[name] = {}
                for fv in var_cfg.names:
                    for action in actions:
                        n_values = len(var_cfg.names)
                        probs = self._rng.dirichlet([1.0] * n_values)
                        step_tables[name][f"{fv}|{action}"] = (var_cfg.names, probs)
            self._within_day.append(step_tables)

    def _sample(self, targets: list[str], probs: np.ndarray) -> str:
        idx = self._rng.choice(len(targets), p=probs)
        return str(targets[idx])

    @override
    def transition(self, state: StateView, action: str) -> dict[str, str]:  # noqa: C901, PLR0912
        updates: dict[str, str] = {}
        stochastic = self._stochastic_factors

        # day_boundary at step 0: sample each factor independently
        if state.step_of_day == 0:
            for name in stochastic:
                if name not in self._day_boundary:
                    continue
                current_val = getattr(state, name)
                if current_val in self._day_boundary[name]:
                    targets, probs = self._day_boundary[name][current_val]
                    updates[name] = self._sample(targets, probs)
                else:
                    logger.warning(
                        "Missing day_boundary key for factor %s value %s",
                        name,
                        current_val,
                    )

        if state.step_of_day >= len(self._within_day):
            msg = (
                f"step_of_day {state.step_of_day} exceeds within_day table count "
                f"{len(self._within_day)}"
            )
            raise IndexError(msg)

        # within_day: sample each stochastic factor independently
        step_tables = self._within_day[state.step_of_day]
        for name in stochastic:
            if name not in step_tables:
                continue
            current_val = getattr(state, name)
            sa_key = f"{current_val}|{action}"
            if sa_key in step_tables[name]:
                targets, probs = step_tables[name][sa_key]
                updates[name] = self._sample(targets, probs)
            else:
                logger.warning(
                    "Missing within_day_%d key for %s=%s, action=%s",
                    state.step_of_day,
                    name,
                    current_val,
                    action,
                )
        return updates

    def save_tables(self, output_dir: str | Path) -> None:
        """Serialize tables to bootstrap-compatible JSON format."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        def _to_json(
            table: dict[str, tuple[list[str], np.ndarray]],
        ) -> dict[str, dict[str, dict[str, float]]]:
            result: dict[str, dict[str, dict[str, float]]] = {}
            for key, (targets, probs) in table.items():
                result[key] = {
                    "_": {
                        t: float(p)
                        for t, p in zip(targets, probs, strict=True)
                    }
                }
            return result

        db_path = out / "day_boundary.json"
        with db_path.open("w", encoding="utf-8") as f:
            json.dump(_to_json(self.day_boundary), f, indent=2)
        logger.info("Saved day_boundary.json with %d entries", len(self.day_boundary))

        flat_wd = self.within_day
        for step_idx, flat_table in enumerate(flat_wd):
            wd_path = out / f"within_day_{step_idx}.json"
            with wd_path.open("w", encoding="utf-8") as f:
                json.dump(_to_json(flat_table), f, indent=2)
            logger.info(
                "Saved within_day_%d.json with %d entries",
                step_idx,
                len(flat_table),
            )


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY.register("random_sa", RandomTransitionSA)
