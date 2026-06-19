"""Run 1b_states_only experiments: all MVP agents, 50 seeds.

Uses Environment directly to capture steps/weight evolution.
"""
import logging
import numpy as np

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.config.schemas import AgentConfig
from rl_health_interventions.environment import Environment
from rl_health_interventions.agents import derive_agent_seed, make as make_agent

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
    """Run one episode, return (total_reward, final_weight, final_steps)."""
    agent_cfg = AgentConfig.model_validate(agent_cfg_dict)
    kwargs = {k: v for k, v in agent_cfg.model_dump().items() if v is not None and k != "type"}
    kwargs["actions"] = config.actions
    kwargs["seed"] = derive_agent_seed(seed, agent_index=0)
    # RandomAgent only accepts actions and seed — filter incompatible kwargs
    import inspect
    from rl_health_interventions.agents import REGISTRY
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

    return total_reward, state.weight, state.steps


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config("docs/initial_experiments/configs/1b_states_only.yaml")
    n_seeds = 50

    print(f"MDP: {config.episode_days} days x {config.steps_per_day} steps")
    print(f"Seeds: {n_seeds}\n")

    results = {}
    for label, agent_dict in AGENT_VARIANTS:
        rewards, wts, steps_arr = [], [], []
        for seed in range(1, n_seeds + 1):
            rw, wt, st = run_episode_capture(config, agent_dict, seed)
            rewards.append(rw)
            if wt is not None:
                wts.append(wt)
            if st is not None:
                steps_arr.append(st)
        r = {
            "total_mean": np.mean(rewards),
            "total_std": np.std(rewards),
            "step_mean": np.mean(rewards) / (config.episode_days * config.steps_per_day),
            "w_mean": np.mean(wts) if wts else None,
            "w_std": np.std(wts) if wts else None,
            "s_mean": np.mean(steps_arr) if steps_arr else None,
            "s_std": np.std(steps_arr) if steps_arr else None,
        }
        results[label] = r
        w_str = f"{r['w_mean']:.2f}+/-{r['w_std']:.2f}" if r['w_mean'] else "N/A"
        s_str = f"{r['s_mean']:.0f}+/-{r['s_std']:.0f}" if r['s_mean'] else "N/A"
        print(f"{label}: reward={r['total_mean']:.1f}+/-{r['total_std']:.1f} weight={w_str} steps={s_str}")

    # LaTeX
    print("\n% --- LATEX TABLE (copy into initial_experiments.tex) ---")
    print("\\begin{tabular}{lrrrr}")
    print("\\toprule")
    print("Agent & Total Reward & Per Step & Weight (kg) & Steps \\\\")
    print("\\midrule")
    for label, _ in AGENT_VARIANTS:
        r = results[label]
        if r["w_mean"] is not None:
            print(f"{label} & {r['total_mean']:.1f} $\\pm$ {r['total_std']:.1f} & {r['step_mean']:.4f} & {r['w_mean']:.2f} $\\pm$ {r['w_std']:.2f} & {r['s_mean']:.0f} $\\pm$ {r['s_std']:.0f} \\\\")
        else:
            print(f"{label} & {r['total_mean']:.1f} $\\pm$ {r['total_std']:.1f} & {r['step_mean']:.4f} & --- & --- \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
