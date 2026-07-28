from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_health_interventions.config.schemas import MDPConfig

logger = logging.getLogger(__name__)

_PROBABILITY_EPSILON = 1e-6


class TableValidator:
    """Validates transition-table JSON structure and probability distributions.

    Responsibility: validate that a parsed JSON dict conforms to the
    required schema — required keys present, ``next_state_probs`` for
    every stochastic factor, and each distribution sums to 1.0 with
    no out-of-range probabilities.
    """

    def validate(self, data: dict, config: MDPConfig) -> None:
        """Validate a single table JSON dict against the MDP config.

        Raises ``ValueError`` with a semicolon-separated message if
        validation fails.
        """
        errors: list[str] = []

        if "transitions" not in data:
            raise ValueError("Missing 'transitions' key in table data")
        if not isinstance(data["transitions"], list):
            raise ValueError("'transitions' must be a list")

        for entry in data["transitions"]:
            errors.extend(self._validate_entry(entry, config))

        if errors:
            raise ValueError("; ".join(errors))

    def _validate_entry(self, entry: dict, config: MDPConfig) -> list[str]:
        missing = self._check_entry_keys(entry)
        if missing:
            return [missing]

        state = entry["state"]
        action = entry["action"]
        next_state_probs = entry["next_state_probs"]

        errs: list[str] = self._check_field_types(state, action, next_state_probs)
        if errs:
            return errs

        config_vars = set(config.state.variables)
        if not set(state.keys()).issubset(config_vars):
            return []

        errs.extend(self._check_stochastic_factors(next_state_probs, config))
        errs.extend(self._check_probabilities(next_state_probs))
        return errs

    @staticmethod
    def _check_field_types(
        state: object,
        action: object,
        next_state_probs: object,
    ) -> list[str]:
        errs: list[str] = []
        if not isinstance(state, dict):
            errs.append("'state' must be a dict")
        if not isinstance(action, str):
            errs.append(f"Action must be a string, got {type(action).__name__}")
        if not isinstance(next_state_probs, dict):
            errs.append("'next_state_probs' must be a dict")
        return errs

    @staticmethod
    def _check_entry_keys(entry: dict) -> str | None:
        if "state" not in entry:
            return "Missing 'state' key in transition entry"
        if "action" not in entry:
            return "Missing 'action' key in transition entry"
        if "next_state_probs" not in entry:
            return "Missing 'next_state_probs' key in transition entry"
        return None

    @staticmethod
    def _check_stochastic_factors(
        next_state_probs: dict,
        config: MDPConfig,
    ) -> list[str]:
        errs: list[str] = []
        stochastic_factors = [
            n for n, c in config.state.variables.items() if c.advanced is None
        ]
        for factor in stochastic_factors:
            if factor not in next_state_probs:
                errs.append(f"Missing stochastic factor '{factor}' in next_state_probs")
        return errs

    @staticmethod
    def _check_probabilities(next_state_probs: dict) -> list[str]:
        errs: list[str] = []
        for factor, probs in next_state_probs.items():
            if not isinstance(probs, dict):
                errs.append(f"next_state_probs for '{factor}' must be a dict")
                continue
            errs.extend(TableValidator._check_single_distribution(factor, probs))
        return errs

    @staticmethod
    def _check_single_distribution(factor: str, probs: dict) -> list[str]:
        errs: list[str] = []
        for val, p in probs.items():
            if p < 0:
                errs.append(f"Negative probability for {factor}={val}: {p}")
            if p > 1.0:
                errs.append(f"Probability > 1.0 for {factor}={val}: {p}")
        total = sum(probs.values())
        if abs(total - 1.0) > _PROBABILITY_EPSILON:
            errs.append(
                f"Probabilities for factor '{factor}' sum to {total}, expected 1.0"
            )
        return errs
