#!/usr/bin/env python3
"""FM Embedding POC: compare hand-crafted vs NormWear FM context.

Runs 5 agents on the same MDP:
  1. Thompson Sampling (no context) — baseline
  2. Thompson Sampling (hand-crafted: step_bin, sleep)
  3. Thompson Sampling (hand-crafted: all 4 factors)
  4. Synthetic FM embedding context (placeholder)
  5. NormWear FM embedding context (real model)

Reports mean reward per agent over 5 seeds. With a real FM, agent 5
should capture waveform-level patterns invisible to hand-crafted factors.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from rl_health_interventions.agents.fm_bandit import FMContextualBanditAgent
from rl_health_interventions.agents.normwear_bandit import NormWearBanditAgent
from rl_health_interventions.agents.contextual_bandits.thompson_sampling import (
    ThompsonSamplingAgent,
)
from rl_health_interventions.config.loader import load_config
from rl_health_interventions.episode import run_episode

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = _project_root / "docs/experimental_phases/fm_poc/configs/fm_poc.yaml"
N_SEEDS = 5  # Fewer seeds for NormWear (slower per episode)
EPISODE_DAYS = 90
STEPS_PER_DAY = 5


def make_agents() -> dict[str, dict]:
    """Create the 5 agents to compare."""
    actions = ["idle", "movement_suggestion", "goal_reminder", "journal"]
    return {
        "ts_no_context": {
            "cls": ThompsonSamplingAgent,
            "kwargs": {
                "actions": actions,
                "alpha_prior": 1.0,
                "beta_prior": 1.0,
                "seed": 42,
            },
        },
        "ts_handcrafted_2feat": {
            "cls": ThompsonSamplingAgent,
            "kwargs": {
                "actions": actions,
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
                "actions": actions,
                "alpha_prior": 1.0,
                "beta_prior": 1.0,
                "seed": 42,
                "contextual": True,
                "context_features": ["step_bin", "sleep", "day_of_week", "burden"],
            },
        },
        "ts_synthetic_fm": {
            "cls": FMContextualBanditAgent,
            "kwargs": {"actions": actions, "seed": 42, "fm_n_bins": 4},
        },
        "ts_normwear_fm": {
            "cls": NormWearBanditAgent,
            "kwargs": {"actions": actions, "seed": 42, "fm_n_bins": 8, "device": "cpu"},
        },
    }


def run_experiment() -> dict:
    """Run all agents across multiple seeds and collect results."""
    config = load_config(str(CONFIG_PATH))
    agent_specs = make_agents()

    results: dict[str, list[float]] = {name: [] for name in agent_specs}
    action_distributions: dict[str, dict[str, int]] = {name: {} for name in agent_specs}
    timings: dict[str, list[float]] = {name: [] for name in agent_specs}

    for seed in range(N_SEEDS):
        for name, spec in agent_specs.items():
            t0 = time.time()
            agent = spec["cls"](**{**spec["kwargs"], "seed": seed})
            records = run_episode(config, agent, seed=seed)
            elapsed = time.time() - t0
            timings[name].append(elapsed)

            total_reward = sum(r["reward"] for r in records)
            results[name].append(total_reward)

            logger.info(
                "  %s seed=%d reward=%.2f time=%.1fs", name, seed, total_reward, elapsed
            )

            # Track action distribution for last seed
            if seed == N_SEEDS - 1:
                for r in records:
                    action_distributions[name][r["action"]] = (
                        action_distributions[name].get(r["action"], 0) + 1
                    )

    return {
        "results": results,
        "action_distributions": action_distributions,
        "timings": timings,
    }


def summarise(results: dict) -> str:
    """Format results as a readable table."""
    lines = []
    lines.append("=" * 76)
    lines.append("FM EMBEDDING POC — Hand-crafted vs Synthetic FM vs NormWear FM")
    lines.append("=" * 76)
    lines.append(
        f"MDP: 36-state factored | {EPISODE_DAYS} days x {STEPS_PER_DAY} steps/day"
    )
    lines.append(f"Seeds: {N_SEEDS} | Reward: alpha*steps + (1-alpha)*sleep - penalty")
    lines.append("-" * 76)
    lines.append("")
    lines.append(
        f"{'Agent':<28} {'Mean':>8} {'Std':>8} {'Median':>8} {'Time':>8}  Context"
    )
    lines.append("-" * 76)

    context_info = {
        "ts_no_context": "0 (none)",
        "ts_handcrafted_2feat": "2 (step_bin, sleep)",
        "ts_handcrafted_4feat": "4 (all factors)",
        "ts_synthetic_fm": "32 (synthetic FM)",
        "ts_normwear_fm": "32 (NormWear 768d)",
    }

    reward_data = results["results"]
    timing_data = results["timings"]
    for name in reward_data:
        rewards = np.array(reward_data[name])
        times = np.array(timing_data[name])
        ctx = context_info.get(name, "?")
        lines.append(
            f"  {name:<26} {rewards.mean():>8.2f} {rewards.std():>8.2f} "
            f"{np.median(rewards):>8.2f} {times.mean():>7.1f}s  {ctx}"
        )

    lines.append("-" * 76)
    lines.append("")

    # Action distributions for last seed
    lines.append("Action distribution (last seed):")
    for name in results["action_distributions"]:
        dist = results["action_distributions"][name]
        total = sum(dist.values())
        dist_str = " | ".join(
            f"{a}: {100 * dist.get(a, 0) / total:.0f}%"
            for a in ["idle", "movement_suggestion", "goal_reminder", "journal"]
        )
        lines.append(f"  {name:<26} {dist_str}")

    lines.append("")
    lines.append("=" * 76)
    lines.append("INTERPRETATION:")
    lines.append(
        "  - NormWear uses REAL 768-dim encoder embeddings from sensor signals"
    )
    lines.append("  - Different action patterns = FM captures different information")
    lines.append("  - If NormWear shows distinct behaviour from hand-crafted, the")
    lines.append("    embedding encodes physiological patterns discrete factors miss")
    lines.append("=" * 76)

    return "\n".join(lines)


def main() -> None:
    logger.info("Running FM embedding POC with NormWear (%d seeds)...", N_SEEDS)
    logger.info("NOTE: NormWear inference is slow on CPU (~30s/episode)")
    logger.info("")
    results = run_experiment()
    report = summarise(results)
    print(report)

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    serialisable = {
        "results": {k: [float(x) for x in v] for k, v in results["results"].items()},
        "action_distributions": results["action_distributions"],
        "timings": {k: [float(x) for x in v] for k, v in results["timings"].items()},
    }
    with open(output_dir / "fm_poc_normwear_results.json", "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved to {output_dir / 'fm_poc_normwear_results.json'}")


if __name__ == "__main__":
    main()
