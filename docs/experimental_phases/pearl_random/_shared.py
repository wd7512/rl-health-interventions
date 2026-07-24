"""Shared utilities for pearl_random experiment scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_health_interventions.agents import derive_agent_seed
from rl_health_interventions.agents import make as make_agent
from rl_health_interventions.episode import run_episode

_CONFIG_DIR = Path(__file__).resolve().parent / "configs"

_SHORT_NAMES: dict[str, str] = {
    "epsilon_greedy": "RL (EG)",
    "comb_weighted_fixed": "Fixed COM-B",
    "fixed": "Fixed",
    "random": "Random",
}


def resolve_config(
    name: str | None = None, *, default: str = "pearl_random.yaml"
) -> str:
    """Resolve a config name/path for this experiment."""
    name = name or default
    p = Path(name)
    return str(p) if p.is_absolute() else str(_CONFIG_DIR / p)


def agent_label(cfg) -> str:
    """Stable human-readable labels for PEARL 4-arm runs."""
    if cfg.type == "fixed" and cfg.action == "idle":
        return "Control"
    return _SHORT_NAMES.get(cfg.type, cfg.type)


def run_agent(config, agent_cfg, n_seeds: int, agent_index: int) -> np.ndarray:
    """Run one agent variant over n_seeds."""
    exclude = {"type"}
    if not agent_cfg.contextual:
        exclude |= {"contextual", "context_features"}
    base_kwargs = agent_cfg.model_dump(exclude=exclude, exclude_none=True)
    base_kwargs["actions"] = config.action_names

    rewards = []
    for seed in range(1, n_seeds + 1):
        kwargs = base_kwargs.copy()
        kwargs["seed"] = derive_agent_seed(seed, agent_index=agent_index)
        agent = make_agent(agent_cfg.type, **kwargs)
        records = run_episode(config, agent, seed=seed)
        rewards.append([r["reward"] for r in records])
    return np.array(rewards)
