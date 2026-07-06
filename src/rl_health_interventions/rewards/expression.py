from __future__ import annotations

import ast
import operator

from typing_extensions import override

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.rewards._base import RewardHandler
from rl_health_interventions.state import StateView

_ALLOWED_NODES = frozenset(
    {
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.UnaryOp,
        ast.USub,
    }
)

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_UNARY_OPS = {
    ast.USub: operator.neg,
}


class ExpressionParser:
    def __init__(self, formula: str) -> None:
        self._compiled = self._compile(formula)

    def _compile(self, formula: str) -> ast.Expression:
        tree = ast.parse(formula, mode="eval")
        assert isinstance(tree, ast.Expression)
        self._validate(tree)
        return tree

    def _validate(self, node: ast.AST) -> None:
        if type(node) not in _ALLOWED_NODES:
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")
        for child in ast.iter_child_nodes(node):
            self._validate(child)

    def evaluate(self, variables: dict[str, float]) -> float:  # noqa: C901
        def _eval(node: ast.AST) -> float:  # noqa: C901, PLR0912
            if isinstance(node, ast.Constant):
                if not isinstance(node.value, (int, float)):
                    raise ValueError(
                        f"Non-numeric constant: {node.value!r} (expected a number)"
                    )
                return float(node.value)
            if isinstance(node, ast.Name):
                try:
                    return variables[node.id]
                except KeyError:
                    raise ValueError(
                        f"Unknown variable '{node.id}' in formula. "
                        f"Available: {sorted(variables)}"
                    ) from None
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                try:
                    op = _BINARY_OPS[type(node.op)]  # ty:ignore[invalid-argument-type]
                except KeyError:
                    raise ValueError(
                        f"Unsupported binary operator: {type(node.op).__name__}"
                    ) from None
                try:
                    return op(left, right)
                except ZeroDivisionError as err:
                    raise ValueError(
                        f"Division by zero in expression: {left} / {right}"
                    ) from err
            if isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                try:
                    op = _UNARY_OPS[type(node.op)]  # ty:ignore[invalid-argument-type]
                except KeyError:
                    raise ValueError(
                        f"Unsupported unary operator: {type(node.op).__name__}"
                    ) from None
                return op(operand)
            raise ValueError(f"Unexpected node: {type(node).__name__}")

        return _eval(self._compiled.body)


class ExpressionReward(RewardHandler):
    def __init__(self, config: MDPConfig) -> None:
        self._config = config
        self._parser = ExpressionParser(config.reward.formula)

    def _resolve_variables(
        self, state: StateView | str, action: str
    ) -> dict[str, float]:
        result = dict(self._config.reward.constants)
        for name, var_cfg in self._config.reward.variables.items():
            if var_cfg.source == "action":
                result[name] = var_cfg.mapping[action]
            elif var_cfg.source.startswith("state."):
                factor_name = var_cfg.source.split(".", 1)[1]
                if isinstance(state, str):
                    result[name] = var_cfg.mapping[state]
                else:
                    factor_value = getattr(state, factor_name)
                    result[name] = var_cfg.mapping[factor_value]
        return result

    @override
    def reward(
        self, state: StateView | str, action: str, step_idx: int
    ) -> tuple[float, bool]:
        variables = self._resolve_variables(state, action)
        value = self._parser.evaluate(variables)
        mult = self._config.reward_multiplier_by_step
        if mult is not None:
            value *= mult[step_idx]
        return value, False


def register() -> None:
    from rl_health_interventions.rewards import REGISTRY

    REGISTRY.register("expression", ExpressionReward)
