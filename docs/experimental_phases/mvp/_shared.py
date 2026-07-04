"""Shared utilities for MVP experiment scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_health_interventions.agents import derive_agent_seed, make as make_agent
from rl_health_interventions.episode import run_episode

_CONFIG_DIR = Path(__file__).resolve().parent / "configs"

_SHORT_NAMES: dict[str, str] = {
    "thompson_sampling": "TS",
    "epsilon_greedy": "EG",
    "ucb": "UCB",
    "decaying_epsilon_greedy": "D-EG",
    "random": "Random",
}


def resolve_config(name: str | None = None, *, default: str = "mvp_extensions.yaml") -> str:
    """Resolve a config name/path. Absolute paths pass through; bare names look in configs/."""
    name = name or default
    p = Path(name)
    return str(p) if p.is_absolute() else str(_CONFIG_DIR / p)


def agent_label(cfg) -> str:
    """Short label like 'Std TS' or 'Ctx UCB'."""
    if cfg.type == "random":
        return "Random"
    short = _SHORT_NAMES.get(cfg.type, cfg.type)
    prefix = "Ctx" if cfg.contextual else "Std"
    return f"{prefix} {short}"


def run_agent(config, agent_cfg, n_seeds: int) -> np.ndarray:
    """Run one agent variant over n_seeds. Returns per-step rewards (n_seeds, n_steps)."""
    exclude = {"type"}
    if not agent_cfg.contextual:
        exclude |= {"contextual", "context_feature"}
    rewards = []
    for seed in range(1, n_seeds + 1):
        kwargs = agent_cfg.model_dump(exclude=exclude, exclude_none=True)
        kwargs["actions"] = config.action_names
        kwargs["seed"] = derive_agent_seed(seed, agent_index=0)
        agent = make_agent(agent_cfg.type, **kwargs)
        records = run_episode(config, agent, seed=seed)
        rewards.append([r["reward"] for r in records])
    return np.array(rewards)
