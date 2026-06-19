"""Run 1c reward experiments across 50 seeds and all MVP agents."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from rl_health_interventions.agents import derive_agent_seed, make as make_agent
from rl_health_interventions.config.schemas import MDPConfig, RewardWeightsConfig
from rl_health_interventions.environment import Environment

# Agent configs for all MVP agents
AGENT_CONFIGS = [
    {"type": "thompson_sampling", "alpha_prior": 1.0, "beta_prior": 1.0},
    {"type": "random"},
    {"type": "epsilon_greedy", "epsilon": 0.1},
    {"type": "ucb", "c": 1.0},
    {"type": "decaying_epsilon_greedy", "epsilon_start": 0.3, "epsilon_min": 0.05, "decay_steps": 200},
]

# Reward mode configs
REWARD_MODES = {
    "no_bonus": None,
    "flat": RewardWeightsConfig(mode="multi_timescale", delayed_reward_interval=21, delayed_reward_value=10.0),
    "scaled": RewardWeightsConfig(mode="multi_timescale", delayed_reward_interval=21, delayed_reward_value=10.0, delayed_reward_scale=5.0, delayed_reward_threshold=0.6),
}

BASE_STATES = {"sedentary": {"reward": 0.0}, "active": {"reward": 1.0}}
TRANSITION = {
    "type": "rule_based",
    "transition_probabilities": {
        "sedentary": {"nudge": {"active": 0.3, "sedentary": 0.7}, "idle": {"active": 0.1, "sedentary": 0.9}},
        "active": {"nudge": {"active": 0.5, "sedentary": 0.5}, "idle": {"active": 0.6, "sedentary": 0.4}},
    },
}


def run_single(agent_cfg: dict, reward_weights, seed: int) -> tuple[float, int]:
    """Run one episode, return (total_reward, bonus_count)."""
    config = MDPConfig(
        episode_days=90,
        steps_per_day=5,
        seed=seed,
        initial_state="sedentary",
        states=BASE_STATES,
        actions=["nudge", "idle"],
        transition_model=TRANSITION,
        reward_weights=reward_weights,
    )
    env = Environment(config, seed=seed)
    agent_kwargs = {k: v for k, v in agent_cfg.items() if v is not None and k != "type"}
    agent_kwargs["actions"] = config.actions
    agent_kwargs["seed"] = derive_agent_seed(seed, agent_index=0)
    agent = make_agent(agent_cfg["type"], **agent_kwargs)

    state = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        agent.update(state, action, reward, next_state)
        state = next_state

    return total_reward, 0  # bonus_count not tracked separately


def main():
    n_seeds = 50
    results = {}

    for mode_name, reward_weights in REWARD_MODES.items():
        for agent_cfg in AGENT_CONFIGS:
            agent_type = agent_cfg["type"]
            rewards = []
            for seed in range(n_seeds):
                rew, _ = run_single(agent_cfg, reward_weights, seed)
                rewards.append(rew)
            mean_rew = sum(rewards) / len(rewards)
            std_rew = (sum((r - mean_rew) ** 2 for r in rewards) / len(rewards)) ** 0.5
            key = f"{mode_name}|{agent_type}"
            results[key] = {"mean": mean_rew, "std": std_rew, "min": min(rewards), "max": max(rewards)}

    # Output as JSON for tex generation
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
