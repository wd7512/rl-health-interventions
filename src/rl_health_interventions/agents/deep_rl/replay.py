from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action_idx: int
    reward: float
    next_state: np.ndarray
    done: bool = False


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def append(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Transition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if batch_size > len(self._buffer):
            raise ValueError("batch_size cannot exceed replay size")
        indices = rng.choice(len(self._buffer), size=batch_size, replace=False)
        return [self._buffer[int(i)] for i in indices]
