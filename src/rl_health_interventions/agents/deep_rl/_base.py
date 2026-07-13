from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np


def _validate_dims(dims: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    if not dims:
        raise ValueError("hidden_dim list must be non-empty")
    if any(h <= 0 for h in dims):
        raise ValueError("hidden_dim entries must be > 0")
    return tuple(dims)


def parse_hidden_dims(hidden_dim: int | list[int] | None) -> tuple[int, ...]:
    if hidden_dim is None:
        return (32,)
    if isinstance(hidden_dim, int):
        return _validate_dims((hidden_dim,))
    return _validate_dims(hidden_dim)


def state_to_key(state) -> tuple[str, ...]:
    return tuple(str(v) for v in state.state_key)


def hash_state_vector(state, state_dim: int) -> np.ndarray:
    if state_dim <= 0:
        raise ValueError("state_dim must be > 0")
    key = state_to_key(state)
    vec = np.zeros(state_dim, dtype=np.float64)
    for idx, value in enumerate(key):
        vec[hash((idx, value)) % state_dim] += 1.0
    return vec


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


@dataclass
class ForwardPass:
    activations: list[np.ndarray]
    output: np.ndarray


class MLP:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        seed: int,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        dims = [input_dim, *hidden_dims, output_dim]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in pairwise(dims):
            scale = np.sqrt(2.0 / fan_in)
            self.weights.append(
                self._rng.normal(loc=0.0, scale=scale, size=(fan_in, fan_out))
            )
            self.biases.append(np.zeros(fan_out, dtype=np.float64))

    def copy(self) -> MLP:
        copied = MLP(1, 1, (), seed=0)
        copied.weights = [w.copy() for w in self.weights]
        copied.biases = [b.copy() for b in self.biases]
        return copied

    def sync_from(self, other: MLP) -> None:
        self.weights = [w.copy() for w in other.weights]
        self.biases = [b.copy() for b in other.biases]

    def forward(self, x: np.ndarray) -> ForwardPass:
        a = x.astype(np.float64, copy=False)
        activations = [a]
        for idx, (w, b) in enumerate(zip(self.weights, self.biases, strict=False)):
            z = a @ w + b
            a = np.maximum(z, 0.0) if idx < len(self.weights) - 1 else z
            activations.append(a)
        return ForwardPass(activations=activations, output=activations[-1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x).output

    def backward_output_gradient(
        self, x: np.ndarray, grad_output: np.ndarray, lr: float
    ) -> None:
        forward = self.forward(x)
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]
        delta = grad_output.astype(np.float64, copy=False)

        for layer_idx in reversed(range(len(self.weights))):
            a_prev = forward.activations[layer_idx]
            grads_w[layer_idx] = np.outer(a_prev, delta)
            grads_b[layer_idx] = delta
            if layer_idx > 0:
                delta = self.weights[layer_idx] @ delta
                relu_grad = (forward.activations[layer_idx] > 0).astype(np.float64)
                delta *= relu_grad

        for idx in range(len(self.weights)):
            self.weights[idx] -= lr * grads_w[idx]
            self.biases[idx] -= lr * grads_b[idx]
