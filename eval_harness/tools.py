"""
tools.py
========

The two tools Phase 4 ships on: a calculator and a Python runner.

Both are deliberately offline and deterministic. Web search was dropped
from the shipping scope for exactly that reason -- it needs a paid key,
it rate-limits, and it returns something different every time you ask,
which makes a routing experiment unreproducible.

    calculate(expr)    arithmetic, no network, no eval()
    run_python(code)   subprocess with a timeout

SAFETY -- read before pointing this at a live model
---------------------------------------------------
``run_python`` executes generated code. It runs in a separate process
with a timeout and a scratch working directory, which stops the common
accidents (infinite loops, a stray file written next to your source) and
nothing else. It is NOT a sandbox: the child has the same filesystem and
network access as you do. Same caveat the upstream HumanEval harness
ships with, and the same mitigation applies -- run it in a container if
you are pointing it at anything you have not read.

``calculate`` is a different story on purpose. It never calls eval():
the expression is parsed to an AST and every node is checked against a
whitelist before a number is produced, so a model that emits
``__import__("os").system(...)`` in place of arithmetic gets None back
rather than a shell. This is a trust boundary -- model output flowing
into an evaluator -- and it is the one place in this file worth being
strict rather than lazy.
"""

from __future__ import annotations

import ast
import operator
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

# Whitelist. Anything not in here is not arithmetic and is refused.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Guard against a model asking for 9**9**9 and hanging the process on a
# number with billions of digits. Exponents stay small in word problems.
_MAX_EXPONENT = 128


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"non-numeric constant: {node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"operator not allowed: {op_type.__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if op_type is ast.Pow and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"exponent too large: {right}")
        return _BIN_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"unary operator not allowed: {op_type.__name__}")
        return _UNARY_OPS[op_type](_eval_node(node.operand))

    raise ValueError(f"node not allowed: {type(node).__name__}")


def calculate(expression: str) -> Optional[str]:
    """
    Evaluate an arithmetic expression. Returns the result as a string, or
    None if the input is not plain arithmetic.

    None rather than an exception because the caller is a routing loop
    that has to keep going: a tool that cannot answer is a tool result of
    "no answer", not a crashed run. The record still logs that the tool
    was called.
    """
    if not isinstance(expression, str) or not expression.strip():
        return None

    # Models write "$1,234" and trailing prose punctuation; strip what is
    # unambiguously formatting before refusing the whole expression.
    cleaned = expression.strip().rstrip("=").replace(",", "").replace("$", "").strip()
    if not cleaned:
        return None

    try:
        tree = ast.parse(cleaned, mode="eval")
        value = _eval_node(tree.body)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError,
            OverflowError, MemoryError, RecursionError):
        return None

    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        # 24.0 -> "24", so the grader's string compare against a gold
        # answer of "24" does not fail on formatting alone.
        if value.is_integer():
            return str(int(value))
        return repr(round(value, 10))
    return str(value)


# ---------------------------------------------------------------------------
# Python runner
# ---------------------------------------------------------------------------

@dataclass
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    timed_out: bool = False


def run_python(code: str, timeout: float = 10.0) -> ExecResult:
    """
    Run ``code`` in a separate interpreter with a timeout.

    Read the SAFETY note at the top of this module first: the child
    process is isolated enough to survive an infinite loop, not enough to
    survive malice.
    """
    if not isinstance(code, str) or not code.strip():
        return ExecResult(ok=False, stdout="", stderr="empty program")

    # A scratch cwd so a program that writes files does not litter the
    # repo, and so two runs cannot see each other's leftovers.
    with tempfile.TemporaryDirectory() as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True,
                timeout=timeout, cwd=workdir,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(ok=False, stdout="", stderr=f"timed out after {timeout}s",
                              timed_out=True)
        except (OSError, ValueError) as exc:
            return ExecResult(ok=False, stdout="", stderr=f"failed to launch: {exc}")

    return ExecResult(ok=proc.returncode == 0, stdout=proc.stdout, stderr=proc.stderr)
