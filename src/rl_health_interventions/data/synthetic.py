from __future__ import annotations

import numpy as np

from rl_health_interventions.data.dataset import Dataset


class SyntheticDataGenerator:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def generate(self, n_users: int = 0, n_timesteps: int = 0) -> Dataset:
        rng = np.random.default_rng(self.seed)
        return Dataset(
            user_ids=np.arange(n_users, dtype=np.int64),
            timestamps=np.empty((n_users, n_timesteps), dtype=np.int64),
            features={
                "steps": rng.normal(8000, 2000, size=(n_users, n_timesteps)).astype(
                    np.int64
                ),
            },
            metadata={
                "seed": self.seed,
                "n_users": n_users,
                "n_timesteps": n_timesteps,
            },
        )


def register() -> None:
    from rl_health_interventions.data import REGISTRY

    REGISTRY["synthetic"] = SyntheticDataGenerator
