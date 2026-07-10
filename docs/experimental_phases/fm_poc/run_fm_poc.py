#!/usr/bin/env python3
"""FM Embedding POC: compare hand-crafted vs FM-embedding context.

Runs 4 agents on the same MDP:
  1. Thompson Sampling (no context) — baseline
  2. Thompson Sampling (hand-crafted context: step_bin, sleep)
  3. Thompson Sampling (hand-crafted context: all 4 factors)
  4. Thompson Sampling (FM embedding context) — POC

Reports mean reward per agent over 10 seeds. If FM embeddings capture
useful information, agent 4 should match or beat agents 2-3.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from rl_health_interventions.agents.fm_bandit import FMContextualBanditAgent
from rl_health_interventions.agents.contextual_bandits.thompson_sampling import (
    ThompsonSamplingAgent,
)
from rl_health_interventions.config.loader import load_config
from rl_health_interventions.episode import run_episode

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = _project_root / "docs/experimental_phases/fm_poc/configs/fm_poc.yaml"
N_SEEDS = 10
EPISODE_DAYS = 90
STEPS_PER_DAY = 5


def make_agents() -> dict[str, dict]:
    """Create the 4 agents to compare."""
    return {
        "ts_no_context": {
            "cls": ThompsonSamplingAgent,
            "kwargs": {
                "actions": ["idle", "movement_suggestion", "goal_reminder", "journal"],
                "alpha_prior": 1.0,
                "beta_prior": 1.0,
                "seed": 42,
            },
        },
        "ts_handcrafted_2feat": {
            "cls": ThompsonSamplingAgent,
            "kwargs": {
                "actions": ["idle", "movement_suggestion", "goal_reminder", "journal"],
                "alpha_prior": 1.0,
                "beta_prior": 1.0,
                "seed": 42,
                "contextual": True,
                "context_features": ["step_bin", "sleep"],
            },
        },
        "ts_handcrafted_4feat": {
            "cls": ThompsonSamplingAgent,
            "kwargs": {
                "actions": ["idle", "movement_suggestion", "goal_reminder", "journal"],
                "alpha_prior": 1.0,
                "beta_prior": 1.0,
                "seed": 42,
                "contextual": True,
                "context_features": ["step_bin", "sleep", "day_of_week", "burden"],
            },
        },
        "ts_fm_embedding": {
            "cls": FMContextualBanditAgent,
            "kwargs": {
                "actions": ["idle", "movement_suggestion", "goal_reminder", "journal"],
                "seed": 42,
                "fm_n_bins": 4,
            },
        },
    }


def run_experiment() -> dict:
    """Run all agents across multiple seeds and collect results."""
    config = load_config(str(CONFIG_PATH))
    agent_specs = make_agents()

    results: dict[str, list[float]] = {name: [] for name in agent_specs}
    action_distributions: dict[str, dict[str, int]] = {name: {} for name in agent_specs}

    for seed in range(N_SEEDS):
        for name, spec in agent_specs.items():
            agent = spec["cls"](**{**spec["kwargs"], "seed": seed})
            records = run_episode(config, agent, seed=seed)
            total_reward = sum(r["reward"] for r in records)
            results[name].append(total_reward)

            # Track action distribution for last seed
            if seed == N_SEEDS - 1:
                for r in records:
                    action_distributions[name][r["action"]] = (
                        action_distributions[name].get(r["action"], 0) + 1
                    )

    return {"results": results, "action_distributions": action_distributions}


def summarise(results: dict) -> str:
    """Format results as a readable table."""
    lines = []
    lines.append("=" * 72)
    lines.append("FM EMBEDDING POC — Hand-crafted vs FM context comparison")
    lines.append("=" * 72)
    lines.append(
        f"MDP: 36-state factored | {EPISODE_DAYS} days × {STEPS_PER_DAY} steps/day"
    )
    lines.append(f"Seeds: {N_SEEDS} | Reward: α·steps + (1-α)·sleep − penalty")
    lines.append("-" * 72)
    lines.append("")
    lines.append(f"{'Agent':<30} {'Mean':>8} {'Std':>8} {'Median':>8}  Context dims")
    lines.append("-" * 72)

    context_info = {
        "ts_no_context": "0 (none)",
        "ts_handcrafted_2feat": "2 (step_bin, sleep)",
        "ts_handcrafted_4feat": "4 (all factors)",
        "ts_fm_embedding": "16 (FM embedding)",
    }

    reward_data = results["results"]
    for name in reward_data:
        rewards = np.array(reward_data[name])
        ctx = context_info.get(name, "?")
        lines.append(
            f"  {name:<28} {rewards.mean():>8.2f} {rewards.std():>8.2f} "
            f"{np.median(rewards):>8.2f}  {ctx}"
        )

    lines.append("-" * 72)
    lines.append("")

    # Action distributions for last seed
    lines.append("Action distribution (last seed):")
    lines.append(
        f"  {'Action':<25}",
    )
    for name in results["action_distributions"]:
        lines.append(f"  {name}")
        dist = results["action_distributions"][name]
        total = sum(dist.values())
        for action in ["idle", "movement_suggestion", "goal_reminder", "journal"]:
            count = dist.get(action, 0)
            pct = 100 * count / total if total > 0 else 0
            lines.append(f"    {action:<23} {count:>4} ({pct:>5.1f}%)")
        lines.append("")

    lines.append("=" * 72)
    lines.append("INTERPRETATION:")
    lines.append("  - If FM embedding agent matches hand-crafted → architecture works")
    lines.append("  - If FM agent uses different action patterns → embedding captures")
    lines.append("    different state information than hand-crafted factors")
    lines.append("  - With a real FM (NormWear), the embedding would capture")
    lines.append("    physiological patterns invisible to discrete factors")
    lines.append("=" * 72)

    return "\n".join(lines)


def main() -> None:
    logger.info("Running FM embedding POC experiment (%d seeds)...", N_SEEDS)
    results = run_experiment()
    report = summarise(results)
    print(report)

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    serialisable = {
        "results": {k: [float(x) for x in v] for k, v in results["results"].items()},
        "action_distributions": results["action_distributions"],
    }
    with open(output_dir / "fm_poc_results.json", "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved to {output_dir / 'fm_poc_results.json'}")


if __name__ == "__main__":
    main()
