from __future__ import annotations

import numpy as np
import pytest

from rl_health_interventions.agents.deep_rl._base import (
    MLP,
    ForwardPass,
    hash_state_vector,
    parse_hidden_dims,
    softmax,
    state_to_key,
    validate_gamma,
    validate_hidden_dim,
    validate_lr,
    validate_positive_float,
    validate_positive_int,
    validate_state_dim,
    validate_unit_interval,
)


class FakeState:
    def __init__(self, *values):
        self.state_key = values


class TestParseHiddenDims:
    def test_none_defaults_to_32(self):
        assert parse_hidden_dims(None) == (32,)

    def test_int_wraps_in_tuple(self):
        assert parse_hidden_dims(16) == (16,)

    def test_list_passes_through(self):
        assert parse_hidden_dims([16, 8]) == (16, 8)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            parse_hidden_dims([])

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="> 0"):
            parse_hidden_dims([16, 0])


class TestStateToKey:
    def test_converts_to_strings(self):
        assert state_to_key(FakeState("a", 1)) == ("a", "1")


class TestHashStateVector:
    def test_deterministic(self):
        state = FakeState("sedentary", "low")
        v1 = hash_state_vector(state, 32)
        v2 = hash_state_vector(state, 32)
        np.testing.assert_array_equal(v1, v2)

    def test_deterministic_across_calls(self):
        state = FakeState("active", "high")
        v1 = hash_state_vector(state, 64)
        v2 = hash_state_vector(state, 64)
        np.testing.assert_array_equal(v1, v2)

    def test_different_states_differ(self):
        v1 = hash_state_vector(FakeState("a"), 64)
        v2 = hash_state_vector(FakeState("b"), 64)
        assert not np.array_equal(v1, v2)

    def test_nonzero_entries(self):
        v = hash_state_vector(FakeState("a", "b"), 32)
        assert np.sum(v) == 2.0

    def test_invalid_dim_raises(self):
        with pytest.raises(ValueError, match="state_dim"):
            hash_state_vector(FakeState("a"), 0)


class TestSoftmax:
    def test_sums_to_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(softmax(v).sum() - 1.0) < 1e-10

    def test_all_positive(self):
        v = np.array([-1.0, 0.0, 1.0])
        assert np.all(softmax(v) > 0)

    def test_large_values_no_overflow(self):
        v = np.array([1000.0, 1001.0, 1002.0])
        s = softmax(v)
        assert np.all(np.isfinite(s))
        assert abs(s.sum() - 1.0) < 1e-10

    def test_uniform_input(self):
        v = np.array([5.0, 5.0, 5.0])
        np.testing.assert_allclose(softmax(v), [1 / 3, 1 / 3, 1 / 3], atol=1e-10)


class TestMLPForward:
    def test_single_linear_layer(self):
        mlp = MLP(input_dim=2, output_dim=1, hidden_dims=(), seed=0)
        w = np.array([[1.0], [2.0]])
        b = np.array([0.5])
        mlp.weights = [w]
        mlp.biases = [b]
        x = np.array([3.0, 4.0])
        expected = x @ w + b
        np.testing.assert_allclose(mlp.predict(x), expected)

    def test_relu_hidden_zeros_negative(self):
        mlp = MLP(input_dim=2, output_dim=1, hidden_dims=(2,), seed=0)
        mlp.weights = [np.eye(2), np.ones((2, 1))]
        mlp.biases = [np.array([-10.0, 0.0]), np.array([0.0])]
        x = np.array([1.0, 1.0])
        result = mlp.predict(x)
        assert result.shape == (1,)

    def test_forward_returns_activations(self):
        mlp = MLP(input_dim=2, output_dim=1, hidden_dims=(3,), seed=0)
        x = np.array([1.0, 2.0])
        fwd = mlp.forward(x)
        assert isinstance(fwd, ForwardPass)
        assert len(fwd.activations) == 3
        np.testing.assert_array_equal(fwd.activations[0], x)
        np.testing.assert_array_equal(fwd.output, fwd.activations[-1])


class TestMLPBackward:
    def _numerical_gradient(self, mlp, x, grad_output, eps=1e-5):
        analytical = []
        for layer in range(len(mlp.weights)):
            gw = np.zeros_like(mlp.weights[layer])
            gb = np.zeros_like(mlp.biases[layer])
            for i in range(mlp.weights[layer].shape[0]):
                for j in range(mlp.weights[layer].shape[1]):
                    mlp.weights[layer][i, j] += eps
                    fp = mlp.predict(x)
                    mlp.weights[layer][i, j] -= 2 * eps
                    fm = mlp.predict(x)
                    mlp.weights[layer][i, j] += eps
                    gw[i, j] = np.sum(grad_output * (fp - fm)) / (2 * eps)
            for i in range(mlp.biases[layer].shape[0]):
                mlp.biases[layer][i] += eps
                fp = mlp.predict(x)
                mlp.biases[layer][i] -= 2 * eps
                fm = mlp.predict(x)
                mlp.biases[layer][i] += eps
                gb[i] = np.sum(grad_output * (fp - fm)) / (2 * eps)
            analytical.append((gw, gb))
        return analytical

    def test_gradient_matches_numerical(self):
        mlp = MLP(input_dim=3, output_dim=2, hidden_dims=(4,), seed=42)
        x = np.array([0.5, -0.3, 0.8])
        grad_output = np.array([1.0, -0.5])
        analytical = self._numerical_gradient(mlp, x, grad_output.copy())
        mlp_copy = mlp.copy()
        lr = 1.0
        mlp.backward_output_gradient(x, grad_output, lr=lr)
        for layer in range(len(mlp.weights)):
            expected_w = mlp_copy.weights[layer] - lr * analytical[layer][0]
            expected_b = mlp_copy.biases[layer] - lr * analytical[layer][1]
            np.testing.assert_allclose(mlp.weights[layer], expected_w, atol=1e-6)
            np.testing.assert_allclose(mlp.biases[layer], expected_b, atol=1e-6)

    def test_linear_only_gradient(self):
        mlp = MLP(input_dim=2, output_dim=1, hidden_dims=(), seed=0)
        mlp.weights = [np.array([[1.0], [2.0]])]
        mlp.biases = [np.array([0.0])]
        x = np.array([3.0, 4.0])
        grad_output = np.array([1.0])
        mlp_copy = mlp.copy()
        mlp.backward_output_gradient(x, grad_output, lr=0.1)
        expected_w = mlp_copy.weights[0] - 0.1 * np.outer(x, grad_output)
        np.testing.assert_allclose(mlp.weights[0], expected_w, atol=1e-10)


class TestMLPCopy:
    def test_independent_weights(self):
        mlp = MLP(input_dim=2, output_dim=1, hidden_dims=(3,), seed=0)
        copied = mlp.copy()
        copied.weights[0][0, 0] = 999.0
        assert mlp.weights[0][0, 0] != 999.0

    def test_same_predictions(self):
        mlp = MLP(input_dim=2, output_dim=1, hidden_dims=(3,), seed=0)
        copied = mlp.copy()
        x = np.array([1.0, 2.0])
        np.testing.assert_array_equal(mlp.predict(x), copied.predict(x))


class TestMLPSyncFrom:
    def test_sync_copies_weights(self):
        src = MLP(input_dim=2, output_dim=1, hidden_dims=(), seed=0)
        dst = MLP(input_dim=2, output_dim=1, hidden_dims=(), seed=1)
        dst.sync_from(src)
        for sw, dw in zip(src.weights, dst.weights, strict=True):
            np.testing.assert_array_equal(sw, dw)
        for sb, db in zip(src.biases, dst.biases, strict=True):
            np.testing.assert_array_equal(sb, db)

    def test_sync_is_independent(self):
        src = MLP(input_dim=2, output_dim=1, hidden_dims=(), seed=0)
        dst = MLP(input_dim=2, output_dim=1, hidden_dims=(), seed=1)
        dst.sync_from(src)
        src.weights[0][0, 0] = 999.0
        assert dst.weights[0][0, 0] != 999.0


class TestMLPFromWeights:
    def test_round_trip(self):
        mlp = MLP(input_dim=3, output_dim=2, hidden_dims=(4,), seed=0)
        restored = MLP.from_weights(
            [w.copy() for w in mlp.weights],
            [b.copy() for b in mlp.biases],
        )
        x = np.array([1.0, -0.5, 0.3])
        np.testing.assert_array_equal(mlp.predict(x), restored.predict(x))


class TestValidationHelpers:
    def test_validate_lr_rejects_zero(self):
        with pytest.raises(ValueError, match="lr"):
            validate_lr(0.0)

    def test_validate_lr_rejects_negative(self):
        with pytest.raises(ValueError, match="lr"):
            validate_lr(-0.01)

    def test_validate_lr_accepts_positive(self):
        validate_lr(0.01)

    def test_validate_gamma_closed(self):
        validate_gamma(0.0)
        validate_gamma(1.0)
        with pytest.raises(ValueError, match="gamma"):
            validate_gamma(-0.1)
        with pytest.raises(ValueError, match="gamma"):
            validate_gamma(1.1)

    def test_validate_gamma_open(self):
        with pytest.raises(ValueError, match="gamma"):
            validate_gamma(0.0, positive=True)
        validate_gamma(0.01, positive=True)

    def test_validate_unit_interval(self):
        validate_unit_interval(0.0, "eps")
        validate_unit_interval(1.0, "eps")
        with pytest.raises(ValueError, match="eps"):
            validate_unit_interval(-0.1, "eps")

    def test_validate_positive_int(self):
        validate_positive_int(1, "n")
        with pytest.raises(ValueError, match="n"):
            validate_positive_int(0, "n")

    def test_validate_positive_float(self):
        validate_positive_float(0.01, "x")
        with pytest.raises(ValueError, match="x"):
            validate_positive_float(0.0, "x")

    def test_validate_state_dim(self):
        validate_state_dim(1)
        with pytest.raises(ValueError, match="state_dim"):
            validate_state_dim(0)

    def test_validate_hidden_dim_int(self):
        validate_hidden_dim(16)
        with pytest.raises(ValueError, match="hidden_dim"):
            validate_hidden_dim(0)

    def test_validate_hidden_dim_list(self):
        validate_hidden_dim([16, 8])
        with pytest.raises(ValueError, match="hidden_dim"):
            validate_hidden_dim([16, 0])
        with pytest.raises(ValueError, match="hidden_dim"):
            validate_hidden_dim([])
