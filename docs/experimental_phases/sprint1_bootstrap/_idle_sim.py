"""Simulate always-idle for each persona to calibrate baselines."""

from __future__ import annotations
import numpy as np
from rl_health_interventions.config.loader import load_config
from rl_health_interventions.environment import Environment

CONFIGS = {
    "Base": "configs/cross/persona_base/sprint1_bootstrap_extensions.yaml",
    "Goal-driven": "configs/cross/persona_goal_driven/sprint1_bootstrap_extensions.yaml",
    "Resistant": "configs/cross/persona_resistant/sprint1_bootstrap_extensions.yaml",
    "Social Responder": "configs/cross/persona_social_responder/sprint1_bootstrap_extensions.yaml",
    "Stable Maintainer": "configs/cross/persona_stable_maintainer/sprint1_bootstrap_extensions.yaml",
    "Initial DeepSeek": "configs/cross/initial_deepseek/sprint1_bootstrap_extensions.yaml",
    "Initial GLM 5.2": "configs/cross/initial_glm5.2/sprint1_bootstrap_extensions.yaml",
}

N_SEEDS = 50
BASE = "docs/experimental_phases/sprint1_bootstrap"

for name, cfg_rel in CONFIGS.items():
    cfg = load_config(f"{BASE}/{cfg_rel}")
    totals = []
    for s in range(N_SEEDS):
        env = Environment(cfg, seed=s)
        state = env.reset()
        ep = 0.0
        done = False
        while not done:
            state, reward, done = env.step("idle")
            ep += reward
        totals.append(ep)
    a = np.array(totals)
    print(f"{name:30s} idle: {a.mean():8.2f} +- {a.std():.2f}  ({a.mean() / 450:.4f}/step)")
