"""
Calculator skill — safe arithmetic expression evaluator.

No eval() is used; instead an AST visitor whitelists only safe node types.
Supports: +, -, *, /, //, **, %, unary -, parentheses, integer/float
literals, and a curated set of math-module functions.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

# ---------------------------------------------------------------------------
# Skill metadata — consumed by SkillRegistry
# ---------------------------------------------------------------------------

SKILL_META = {
    "name": "calculator",
    "description": (
        "Use for math: arithmetic (+, -, *, /), powers (**), square roots (sqrt), "
        "trigonometry (sin, cos, tan, log), and constants (pi, e). "
        "Use this whenever the user needs a numerical calculation."
    ),
    "trigger_patterns": [
        r"\bcalculate\b",
        r"\bcompute\b",
        r"\bevaluate\b",
        r"\bsolve\b",
        r"what\s+is\s+[\d(]",
        r"\bmath\b",
        r"\d+\s*[\+\-\*\/\^]\s*\d+",
        r"sqrt|sin|cos|tan|log|ceil|floor",
    ],
    "version": "1.0.0",
    "author": "eval_framework",
    # Extra metadata used by end-to-end benchmark
    "examples": [
        {"input": "2 + 2", "expected": 4},
        {"input": "sqrt(144)", "expected": 12.0},
        {"input": "2 ** 10", "expected": 1024},
        {"input": "(3 + 4) * 5 - 2", "expected": 33},
        {"input": "sin(0)", "expected": 0.0},
        {"input": "log(math.e)", "expected": 1.0},
    ],
}

# ---------------------------------------------------------------------------
# Allowed math functions / constants
# ---------------------------------------------------------------------------

_ALLOWED_NAMES: dict[str, Any] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "pow": math.pow,
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "round": round,
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    # Allow "math.X" style access
    "math": math,
}

_BINARY_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitXor: operator.xor,  # allow ^ as xor (not power — use ** for that)
}

_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


# ---------------------------------------------------------------------------
# Safe AST evaluator
# ---------------------------------------------------------------------------


class _SafeEvaluator(ast.NodeVisitor):
    """Recursively evaluates a whitelisted subset of Python AST nodes."""

    def visit(self, node: ast.AST) -> Any:  # type: ignore[override]
        method = f"visit_{type(node).__name__}"
        handler = getattr(self, method, None)
        if handler is None:
            raise ValueError(
                f"Unsupported expression node: {type(node).__name__!r}. "
                "Only arithmetic operations and whitelisted math functions are allowed."
            )
        return handler(node)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__!r}")

    # Python < 3.8 compat (Num node)
    def visit_Num(self, node: Any) -> Any:  # type: ignore[no-untyped-def]
        return node.n

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op_fn = _BINARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__!r}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return op_fn(left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__!r}")
        return op_fn(self.visit(node.operand))

    def visit_Call(self, node: ast.Call) -> Any:
        # Resolve function name
        if isinstance(node.func, ast.Name):
            fn_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # e.g. math.sqrt(...)
            if isinstance(node.func.value, ast.Name):
                fn_name = f"{node.func.value.id}.{node.func.attr}"
                # Resolve math.X
                if node.func.value.id == "math":
                    fn_obj = getattr(math, node.func.attr, None)
                    if fn_obj is None:
                        raise ValueError(f"Unknown math function: {fn_name!r}")
                    args = [self.visit(a) for a in node.args]
                    return fn_obj(*args)
            raise ValueError("Only simple function calls are supported (e.g. sqrt(x))")
        else:
            raise ValueError("Only simple function calls are supported (e.g. sqrt(x))")

        fn_obj = _ALLOWED_NAMES.get(fn_name)
        if fn_obj is None or not callable(fn_obj):
            raise ValueError(
                f"Function {fn_name!r} is not allowed. "
                f"Allowed: {[k for k, v in _ALLOWED_NAMES.items() if callable(v)]}"
            )
        args = [self.visit(a) for a in node.args]
        return fn_obj(*args)

    def visit_Name(self, node: ast.Name) -> Any:
        val = _ALLOWED_NAMES.get(node.id)
        if val is None:
            raise ValueError(
                f"Name {node.id!r} is not allowed. "
                f"Allowed names: {list(_ALLOWED_NAMES.keys())}"
            )
        return val


def _evaluate_expression(expr: str) -> Any:
    """Parse and safely evaluate an arithmetic expression string."""
    # Normalise: replace ^ with ** only when used as power (heuristic)
    expr = expr.strip()
    # Strip leading/trailing quotes that might come from a model
    expr = re.sub(r"^['\"]|['\"]$", "", expr)

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Syntax error in expression {expr!r}: {exc}") from exc

    evaluator = _SafeEvaluator()
    result = evaluator.visit(tree)

    if isinstance(result, float):
        if math.isinf(result):
            raise OverflowError("Result is infinite")
        if math.isnan(result):
            raise ValueError("Result is NaN")

    return result


# ---------------------------------------------------------------------------
# Public execute function — called by the SkillRegistry
# ---------------------------------------------------------------------------


def execute(input: Any) -> Any:  # noqa: A002
    """
    Evaluate the arithmetic expression in ``input.query``.

    Accepts a ``SkillInput`` object (from the registry) or a plain dict.
    Returns a ``SkillOutput`` (or dict with the same keys).
    """
    # Support both SkillInput objects and raw dicts (handy in tests)
    if hasattr(input, "query"):
        query: str = input.query
        params: dict = getattr(input, "parameters", {})
    else:
        query = input.get("query", "")
        params = input.get("parameters", {})

    # Allow the expression to be passed directly in parameters["expression"]
    expression = params.get("expression", query)

    try:
        # Import here to keep skill standalone when loaded directly
        from eval_framework.skills.registry import SkillOutput  # noqa: PLC0415
    except ImportError:
        # Fallback: return a plain dict (useful when running skill in isolation)
        try:
            result = _evaluate_expression(expression)
            return {"result": result, "success": True, "error": None,
                    "metadata": {"expression": expression, "type": type(result).__name__}}
        except Exception as exc:
            return {"result": None, "success": False, "error": str(exc), "metadata": {}}

    try:
        result = _evaluate_expression(expression)
        return SkillOutput(
            result=result,
            success=True,
            metadata={"expression": expression, "type": type(result).__name__},
        )
    except ZeroDivisionError:
        return SkillOutput.failure("division by zero", metadata={"expression": expression})
    except OverflowError as exc:
        return SkillOutput.failure(str(exc), metadata={"expression": expression})
    except ValueError as exc:
        return SkillOutput.failure(str(exc), metadata={"expression": expression})
    except Exception as exc:
        return SkillOutput.failure(
            f"Unexpected error: {exc}", metadata={"expression": expression}
        )
