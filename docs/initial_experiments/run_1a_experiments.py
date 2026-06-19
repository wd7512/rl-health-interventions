"""Run 1a_actions_only experiments: all MVP agents, 50 seeds.

6-action config with reward_penalty/burden_penalty metadata.
"""
import logging
import numpy as np

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.config.schemas import AgentConfig
from rl_health_interventions.environment import Environment
from rl_health_interventions.agents import derive_agent_seed, make as make_agent
from rl_health_interventions.agents import REGISTRY
import inspect

AGENT_VARIANTS = [
    ("Standard TS", {"type": "thompson_sampling", "alpha_prior": 1, "beta_prior": 1}),
    ("Contextual TS", {"type": "thompson_sampling", "alpha_prior": 1, "beta_prior": 1, "contextual": True, "context_feature": "activity"}),
    ("Standard EG", {"type": "epsilon_greedy", "epsilon": 0.05}),
    ("Contextual EG", {"type": "epsilon_greedy", "epsilon": 0.05, "contextual": True, "context_feature": "activity"}),
    ("Standard UCB", {"type": "ucb", "c": 0.5}),
    ("Contextual UCB", {"type": "ucb", "c": 0.5, "contextual": True, "context_feature": "activity"}),
    ("Standard DEC", {"type": "decaying_epsilon_greedy", "epsilon_start": 0.2, "epsilon_min": 0.01, "decay_steps": 200}),
    ("Contextual DEC", {"type": "decaying_epsilon_greedy", "epsilon_start": 0.2, "epsilon_min": 0.01, "decay_steps": 200, "contextual": True, "context_feature": "activity"}),
    ("Random", {"type": "random"}),
]


def run_episode_capture(config, agent_cfg_dict, seed):
    """Run one episode, return (total_reward, final_steps)."""
    agent_cfg = AgentConfig.model_validate(agent_cfg_dict)
    kwargs = {k: v for k, v in agent_cfg.model_dump().items() if v is not None and k != "type"}
    kwargs["actions"] = config.action_names
    kwargs["seed"] = derive_agent_seed(seed, agent_index=0)
    # Filter to only kwargs the agent accepts
    sig = inspect.signature(REGISTRY[agent_cfg.type].__init__)
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    agent = make_agent(agent_cfg.type, **kwargs)

    env = Environment(config, seed=seed)
    state = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        total_reward += reward
        state = next_state

    return total_reward


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config("docs/initial_experiments/configs/1a_actions_only.yaml")
    n_seeds = 50

    print(f"MDP: {config.episode_days} days x {config.steps_per_day} steps")
    print(f"Actions: {config.action_names}")
    print(f"Seeds: {n_seeds}\n")

    results = {}
    for label, agent_dict in AGENT_VARIANTS:
        rewards = []
        for seed in range(1, n_seeds + 1):
            rw = run_episode_capture(config, agent_dict, seed)
            rewards.append(rw)
        r = {
            "total_mean": np.mean(rewards),
            "total_std": np.std(rewards),
            "step_mean": np.mean(rewards) / (config.episode_days * config.steps_per_day),
        }
        results[label] = r
        print(f"{label}: reward={r['total_mean']:.1f}+/-{r['total_std']:.1f}")

    # LaTeX
    print("\n% --- LATEX TABLE (copy into initial_experiments.tex) ---")
    print("\\begin{tabular}{lrrr}")
    print("\\toprule")
    print("Agent & Total Reward & Per Step & Active Fraction \\\\")
    print("\\midrule")
    for label, _ in AGENT_VARIANTS:
        r = results[label]
        print(f"{label} & {r['total_mean']:.1f} $\\pm$ {r['total_std']:.1f} & {r['step_mean']:.4f} & --- \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
