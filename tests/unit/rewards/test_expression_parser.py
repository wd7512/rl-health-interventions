import pytest
from rl_health_interventions.rewards.expression import ExpressionParser


def test_simple_constant():
    p = ExpressionParser("42")
    assert p.evaluate({}) == 42.0


def test_variable_lookup():
    p = ExpressionParser("x")
    assert p.evaluate({"x": 3.5}) == 3.5


def test_simple_arithmetic():
    p = ExpressionParser("x + y * 2")
    assert p.evaluate({"x": 1.0, "y": 3.0}) == 7.0


def test_subtraction():
    p = ExpressionParser("x - y")
    assert p.evaluate({"x": 10.0, "y": 3.0}) == 7.0


def test_division():
    p = ExpressionParser("x / y")
    assert p.evaluate({"x": 10.0, "y": 2.0}) == 5.0


def test_unary_minus():
    p = ExpressionParser("-x")
    assert p.evaluate({"x": 5.0}) == -5.0


def test_complex_expression():
    p = ExpressionParser("alpha * step_bin + (1 - alpha) * sleep - penalty")
    variables = {"alpha": 0.9, "step_bin": 1.0, "sleep": 0.5, "penalty": 0.1}
    expected = 0.9 * 1.0 + (1 - 0.9) * 0.5 - 0.1
    assert p.evaluate(variables) == pytest.approx(expected)


def test_missing_variable_raises():
    p = ExpressionParser("x + y")
    with pytest.raises(ValueError, match="Unknown variable"):
        p.evaluate({"x": 1.0})


def test_unsafe_node_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        ExpressionParser("__import__('os')")


def test_unsafe_function_call_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        ExpressionParser("max(1, 2)")


def test_unsafe_attribute_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        ExpressionParser("x.__class__")


def test_unsafe_binary_op_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        ExpressionParser("x ** 2")
