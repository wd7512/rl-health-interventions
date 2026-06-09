from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Dataset:
    user_ids: np.ndarray
    timestamps: np.ndarray
    features: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def validate(self) -> None:
        n_users = self.user_ids.shape[0]
        if self.timestamps.shape[0] != n_users:
            raise ValueError(
                f"timestamps shape {self.timestamps.shape} does not match "
                f"user_ids shape ({n_users},)"
            )
        for name, arr in self.features.items():
            if arr.shape[0] != n_users:
                raise ValueError(
                    f"feature '{name}' shape {arr.shape} does not match "
                    f"user_ids shape ({n_users},)"
                )
