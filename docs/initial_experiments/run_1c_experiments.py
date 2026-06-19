"""Run 1c reward experiments across 50 seeds and all MVP agents."""
from __future__ import annotations

import json
import logging
import numpy as np

from rl_health_interventions.agents import derive_agent_seed, make as make_agent
from rl_health_interventions.config.schemas import MDPConfig, RewardWeightsConfig
from rl_health_interventions.environment import Environment

logger = logging.getLogger(__name__)

# All MVP agents: standard + contextual variants (matching run_1b_benchmark.py)
AGENT_CONFIGS = [
    {"type": "thompson_sampling", "alpha_prior": 1.0, "beta_prior": 1.0},
    {"type": "thompson_sampling", "alpha_prior": 1.0, "beta_prior": 1.0,
     "contextual": True, "context_feature": "activity"},
    {"type": "epsilon_greedy", "epsilon": 0.05},
    {"type": "epsilon_greedy", "epsilon": 0.05,
     "contextual": True, "context_feature": "activity"},
    {"type": "ucb", "c": 0.5},
    {"type": "ucb", "c": 0.5,
     "contextual": True, "context_feature": "activity"},
    {"type": "decaying_epsilon_greedy", "epsilon_start": 0.2,
     "epsilon_min": 0.01, "decay_steps": 200},
    {"type": "decaying_epsilon_greedy", "epsilon_start": 0.2,
     "epsilon_min": 0.01, "decay_steps": 200,
     "contextual": True, "context_feature": "activity"},
    {"type": "random"},
]

AGENT_LABELS = [
    "Thompson Sampling",
    "Contextual TS",
    "Epsilon-greedy",
    "Contextual EG",
    "UCB",
    "Contextual UCB",
    "Decaying Eps-greedy",
    "Contextual DEC",
    "Random",
]

# Reward mode configs
REWARD_MODES = {
    "no_bonus": None,
    "flat": RewardWeightsConfig(
        mode="multi_timescale",
        delayed_reward_interval=21,
        delayed_reward_value=10.0,
    ),
    "scaled": RewardWeightsConfig(
        mode="multi_timescale",
        delayed_reward_interval=21,
        delayed_reward_value=10.0,
        delayed_reward_scale=5.0,
        delayed_reward_threshold=0.6,
    ),
}

BASE_STATES = {"sedentary": {"reward": 0.0}, "active": {"reward": 1.0}}
TRANSITION = {
    "type": "rule_based",
    "transition_probabilities": {
        "sedentary": {
            "nudge": {"active": 0.3, "sedentary": 0.7},
            "idle": {"active": 0.1, "sedentary": 0.9},
        },
        "active": {
            "nudge": {"active": 0.5, "sedentary": 0.5},
            "idle": {"active": 0.6, "sedentary": 0.4},
        },
    },
}


def run_single(agent_cfg: dict, reward_weights, seed: int) -> float:
    """Run one episode, return total_reward."""
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
    agent_kwargs = {
        k: v for k, v in agent_cfg.items()
        if v is not None and k not in ("type", "contextual", "context_feature")
    }
    # Match 1b pattern: pass config.actions (list of action names)
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

    return total_reward


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n_seeds = 50
    results = {}

    for mode_name, reward_weights in REWARD_MODES.items():
        for i, (agent_cfg, label) in enumerate(
            zip(AGENT_CONFIGS, AGENT_LABELS)
        ):
            rewards = []
            for seed in range(n_seeds):
                rew = run_single(agent_cfg, reward_weights, seed)
                rewards.append(rew)
            mean_rew = np.mean(rewards)
            std_rew = np.std(rewards)
            key = f"{mode_name}|{label}"
            results[key] = {
                "mean": round(float(mean_rew), 1),
                "std": round(float(std_rew), 1),
            }
            logger.info(
                "%s | %s: %.1f ± %.1f",
                mode_name, label, mean_rew, std_rew,
            )

    # Output as JSON for tex generation
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
