"""Shared utilities for PEARL Constitution validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from rl_health_interventions.agents import derive_agent_seed
from rl_health_interventions.agents import make as make_agent
from rl_health_interventions.config.loader import load_config
from rl_health_interventions.config.schemas import AgentConfig, MDPConfig
from rl_health_interventions.episode import run_episode

logger = logging.getLogger(__name__)

# Step-bin midpoint mapping (from environment.py)
_STEP_BIN_MIDPOINT: dict[str, int] = {
    "inactive": 400,
    "moderate": 1200,
    "active": 2000,
}

# Known persona table directories relative to repo root
PERSONA_TABLE_DIRS: dict[str, str] = {
    "base": "tables/persona/base_deepseek-v4-flash",
    "goal_driven": "tables/persona/goal_driven_deepseek-v4-flash",
    "resistant": "tables/persona/resistant_deepseek-v4-flash",
    "social_responder": "tables/persona/social_responder_deepseek-v4-flash",
    "stable_maintainer": "tables/persona/stable_maintainer_deepseek-v4-flash",
}

# Arm names in order matching config agents list
ARM_NAMES: list[str] = ["control", "random", "fixed", "rl"]

# PEARL reference data path
_REFERENCE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "docs"
    / "research"
    / "reference"
    / "pearl_reference.json"
)


# ── Reference data ──────────────────────────────────────────────────────────


def load_reference() -> dict[str, Any]:
    """Load PEARL reference statistics from JSON."""
    path = _REFERENCE_PATH
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


# ── Daily steps computation ─────────────────────────────────────────────────


def compute_daily_steps(records: list[dict]) -> np.ndarray:
    """Compute daily step totals from trajectory records.

    Parameters
    ----------
    records : list[dict]
        Trajectory records from ``run_episode()``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_days,)`` with total steps per day.
    """
    records_by_day: dict[int, list[dict]] = {}
    for rec in records:
        records_by_day.setdefault(rec["day"], []).append(rec)

    n_days = max(records_by_day.keys()) + 1 if records_by_day else 0
    daily_steps = np.zeros(n_days, dtype=np.float64)
    for day, recs in records_by_day.items():
        total = sum(
            _STEP_BIN_MIDPOINT.get(r.get("step_bin", "inactive"), 0) for r in recs
        )
        daily_steps[day] = total
    return daily_steps


# ── Multi-seed arm runner ───────────────────────────────────────────────────


def run_arm_trajectories(
    config: MDPConfig,
    agent_cfg: AgentConfig,
    n_seeds: int,
) -> list[list[dict]]:
    """Run one agent arm across multiple seeds, returning full trajectory records.

    Parameters
    ----------
    config : MDPConfig
        Loaded MDP configuration.
    agent_cfg : AgentConfig
        Agent configuration for one arm.
    n_seeds : int
        Number of seeds to run.

    Returns
    -------
    list[list[dict]]
        Outer list: seeds. Inner list: trajectory records for that seed.
    """
    exclude = {"type"}
    if not agent_cfg.contextual:
        exclude |= {"contextual", "context_features"}
    base_kwargs = agent_cfg.model_dump(exclude=exclude, exclude_none=True)
    base_kwargs["actions"] = config.action_names

    all_trajectories: list[list[dict]] = []
    for seed in range(1, n_seeds + 1):
        kwargs = base_kwargs.copy()
        kwargs["seed"] = derive_agent_seed(seed, agent_index=0)
        agent = make_agent(agent_cfg.type, **kwargs)
        records = run_episode(config, agent, seed=seed)
        all_trajectories.append(records)
    return all_trajectories


def run_all_arms(
    config: MDPConfig,
    n_seeds: int = 10,
) -> dict[str, list[list[dict]]]:
    """Run all 4 PEARL arms and return trajectory data.

    Parameters
    ----------
    config : MDPConfig
        Loaded MDP configuration (must have exactly 4 agents).
    n_seeds : int
        Number of seeds per arm.

    Returns
    -------
    dict[str, list[list[dict]]]
        Mapping from arm name to list of seed trajectories.
    """
    results: dict[str, list[list[dict]]] = {}
    for idx, agent_cfg in enumerate(config.agents):
        arm_name = ARM_NAMES[idx]
        logger.info("Running arm %d/%d: %s ...", idx + 1, len(config.agents), arm_name)
        trajectories = run_arm_trajectories(config, agent_cfg, n_seeds)
        results[arm_name] = trajectories
    return results


# ── Statistical helpers ─────────────────────────────────────────────────────


def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
) -> float:
    """Cohen's d effect size between two independent groups."""
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled)


def icc_1way_random(
    data: np.ndarray,
) -> float:
    """Intraclass correlation coefficient (ICC(1,1)) from a one-way random effects model.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(n_targets, n_raters)`` or ``(n_subjects, n_measurements)``.

    Returns
    -------
    float
        ICC estimate.
    """
    n, k = data.shape
    # Between-target mean square
    target_means = np.mean(data, axis=1)
    grand_mean = np.mean(target_means)
    ms_between = k / (n - 1) * np.sum((target_means - grand_mean) ** 2)

    # Within-target (error) mean square
    ms_within = 1 / (n * (k - 1)) * np.sum((data - target_means[:, None]) ** 2)

    if ms_between == 0:
        return 0.0
    return float((ms_between - ms_within) / (ms_between + (k - 1) * ms_within))


def anova_oneway(
    groups: list[np.ndarray],
) -> tuple[float, float]:
    """One-way ANOVA returning (F, p-value)."""
    f_stat, p_val = sp_stats.f_oneway(*groups)
    return float(f_stat), float(p_val)


def eta_squared(
    groups: list[np.ndarray],
) -> float:
    """Eta-squared effect size for one-way ANOVA."""
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ss_total = np.sum((all_data - grand_mean) ** 2)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    if ss_total == 0:
        return 0.0
    return float(ss_between / ss_total)


# ── Config helpers ──────────────────────────────────────────────────────────


def load_constitution_config(
    persona: str = "base",
    config_path: str | Path = "config/pearl_constitution.yaml",
) -> MDPConfig:
    """Load the PEARL constitution config, optionally switching persona.

    Parameters
    ----------
    persona : str
        Persona name from ``PERSONA_TABLE_DIRS`` keys.
    config_path : str or Path
        Path to the YAML config file.

    Returns
    -------
    MDPConfig
        Loaded and validated config with persona-specific table_dir.
    """
    config = load_config(config_path)

    # Override table_dir for persona mixture
    table_rel = PERSONA_TABLE_DIRS.get(persona)
    if table_rel is not None:
        config_path_resolved = Path(config_path).resolve()
        repo_root = config_path_resolved.parent.parent  # config/ -> repo root
        config.transition_model.table_dir = str(repo_root / table_rel)
        logger.info(
            "Using persona '%s': table_dir=%s",
            persona,
            config.transition_model.table_dir,
        )
    else:
        logger.warning("Unknown persona '%s', keeping default table_dir", persona)

    return config


# ── Arm-level aggregate helpers ─────────────────────────────────────────────


def compute_arm_daily_steps(
    trajectories: dict[str, list[list[dict]]],
) -> dict[str, np.ndarray]:
    """Compute daily steps for each arm across all seeds.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from arm name to array of shape ``(n_seeds, n_days)``.
    """
    result: dict[str, np.ndarray] = {}
    for arm_name, seed_trajs in trajectories.items():
        daily_list = [compute_daily_steps(traj) for traj in seed_trajs]
        # Ensure all same length
        min_len = min(len(d) for d in daily_list)
        result[arm_name] = np.array([d[:min_len] for d in daily_list])
    return result


def mean_daily_steps_by_arm(
    daily_steps: dict[str, np.ndarray],
    day_start: int = 0,
    day_end: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-seed mean daily steps over a window for each arm.

    Parameters
    ----------
    daily_steps : dict[str, np.ndarray]
        Mapping from arm name to ``(n_seeds, n_days)``.
    day_start : int
        Start day (inclusive).
    day_end : int or None
        End day (exclusive). None means all days.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from arm name to ``(n_seeds,)`` of mean daily steps.
    """
    result: dict[str, np.ndarray] = {}
    for arm, data in daily_steps.items():
        window = data[:, day_start:day_end]
        result[arm] = np.mean(window, axis=1)
    return result


def one_month_days(steps_per_day: int = 5) -> int:
    """Number of days in approximately one month (30 days)."""
    return 30


def two_month_days(steps_per_day: int = 5) -> int:
    """Number of days in two months (60 days)."""
    return 60


# ── Result formatting ───────────────────────────────────────────────────────


def format_check_result(
    check_id: str,
    name: str,
    passed: bool,
    detail: str = "",
    tier: int = 1,
) -> dict[str, Any]:
    """Format a single check result as a dict."""
    return {
        "tier": tier,
        "check_id": check_id,
        "name": name,
        "passed": passed,
        "detail": detail,
    }


def format_matrix(
    results: list[dict[str, Any]],
) -> str:
    """Format a summary matrix of all check results.

    Returns
    -------
    str
        Multi-line string with a summary table.
    """
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  PEARL Constitution — Validation Summary Matrix")
    lines.append("=" * 72)
    lines.append("")

    for tier in range(1, 5):
        tier_results = [r for r in results if r.get("tier") == tier]
        if not tier_results:
            continue
        lines.append(f"── Tier {tier} ──")
        for r in tier_results:
            status = "✅ PASS" if r["passed"] else "❌ FAIL"
            lines.append(
                f"  {r['check_id']:8s} {status}  | "
                f"{r['name']:40s} | {r.get('detail', '')}"
            )
        lines.append("")

    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)
    lines.append(f"Total: {n_pass}/{n_total} checks passed")
    lines.append("=" * 72)
    return "\n".join(lines)
