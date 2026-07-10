#!/usr/bin/env python3
"""Experiment: FM-guided dynamic action space.

Compares:
1. ts_4action: Thompson Sampling with 4 fixed actions (baseline)
2. ts_15action: Thompson Sampling with 15 actions (flat, no FM)
3. fm_guided_15action: FM-guided 2-stage selection (archetype → action)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.episode import run_episode
from rl_health_interventions.agents.contextual_bandits.thompson_sampling import (
    ThompsonSamplingAgent,
)


def _make_fm_guided():
    from rl_health_interventions.agents.fm_guided_bandit import FMGuidedBanditAgent
    return FMGuidedBanditAgent()


def run_comparison(n_seeds: int = 5, output_dir: str | None = None):
    """Run the full comparison experiment."""
    config_15 = load_config("docs/experimental_phases/fm_poc/configs/fm_action_space.yaml")
    config_4 = load_config("docs/experimental_phases/fm_poc/configs/fm_poc.yaml")

    agents_spec = [
        ("ts_4action", lambda: ThompsonSamplingAgent(
            actions=list(config_4.actions.keys()),
            seed=42,
            alpha_prior=1.0,
            beta_prior=1.0,
        ), config_4),
        ("ts_15action", lambda: ThompsonSamplingAgent(
            actions=list(config_15.actions.keys()),
            seed=42,
            alpha_prior=1.0,
            beta_prior=1.0,
        ), config_15),
        ("fm_guided_15action", _make_fm_guided, config_15),
    ]

    results = []
    print()
    print("=" * 78)
    print("FM-Guided Dynamic Action Space Experiment")
    print("=" * 78)
    print(f"MDP: 36-state factored | 90 days x 5 steps/day")
    print(f"Seeds: {n_seeds}")
    print("-" * 78)
    print()

    for agent_name, make_agent, cfg in agents_spec:
        rewards = []
        actions_dist: dict[str, int] = {}
        for seed in range(n_seeds):
            agent = make_agent()
            t0 = time.time()
            records = run_episode(cfg, agent, seed=seed)
            elapsed = time.time() - t0
            total_reward = sum(r["reward"] for r in records)
            rewards.append(total_reward)

            for r in records:
                a = r["action"]
                actions_dist[a] = actions_dist.get(a, 0) + 1

            print(f"  {agent_name:30s} seed={seed} reward={total_reward:7.2f} time={elapsed:5.1f}s")

        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards))
        median_r = float(np.median(rewards))

        total_actions = sum(actions_dist.values())
        dist_str = " | ".join(
            f"{a}: {100*c/total_actions:.0f}%"
            for a, c in sorted(actions_dist.items(), key=lambda x: -x[1])[:6]
        )

        results.append({
            "agent": agent_name,
            "mean_reward": round(mean_r, 2),
            "std_reward": round(std_r, 2),
            "median_reward": round(median_r, 2),
            "action_distribution": {
                a: round(100 * c / total_actions, 1)
                for a, c in sorted(actions_dist.items(), key=lambda x: -x[1])
            },
        })

        print()
        print(f"  {agent_name:30s} mean={mean_r:.2f} +/- {std_r:.2f}  median={median_r:.2f}")
        print(f"  Action dist: {dist_str}")
        print()

    # Summary table
    print("-" * 78)
    print(f"{'Agent':35s} {'Mean':>8s} {'Std':>8s} {'Median':>8s}")
    print("-" * 78)
    for r in results:
        print(f"  {r['agent']:33s} {r['mean_reward']:8.2f} {r['std_reward']:8.2f} {r['median_reward']:8.2f}")
    print("-" * 78)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "fm_action_space_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out / 'fm_action_space_results.json'}")

    return results


if __name__ == "__main__":
    run_comparison(
        n_seeds=5,
        output_dir="docs/experimental_phases/fm_poc/results",
    )
