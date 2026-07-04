import pytest
from rl_health_interventions.rewards.expression import ExpressionParser


@pytest.mark.parametrize(
    "expr,variables,expected",
    [
        ("42", {}, 42.0),
        ("x", {"x": 3.5}, 3.5),
        ("x + y * 2", {"x": 1.0, "y": 3.0}, 7.0),
        ("x - y", {"x": 10.0, "y": 3.0}, 7.0),
        ("x / y", {"x": 10.0, "y": 2.0}, 5.0),
        ("-x", {"x": 5.0}, -5.0),
        pytest.param(
            "alpha * step_bin + (1 - alpha) * sleep - penalty",
            {"alpha": 0.9, "step_bin": 1.0, "sleep": 0.5, "penalty": 0.1},
            0.9 * 1.0 + (1 - 0.9) * 0.5 - 0.1,
            id="complex-expression",
        ),
    ],
)
def test_evaluate(expr, variables, expected):
    assert ExpressionParser(expr).evaluate(variables) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expr,evaluate_vars,expected_match",
    [
        ("__import__('os')", {}, "Unsupported"),
        ("max(1, 2)", {}, "Unsupported"),
        ("x.__class__", {}, "Unsupported"),
        ("x ** 2", {}, "Unsupported"),
        ("x + y", {"x": 1.0}, "Unknown variable"),
    ],
)
def test_expression_errors(expr, evaluate_vars, expected_match):
    if evaluate_vars:
        p = ExpressionParser(expr)
        with pytest.raises(ValueError, match=expected_match):
            p.evaluate(evaluate_vars)
    else:
        with pytest.raises(ValueError, match=expected_match):
            ExpressionParser(expr)
