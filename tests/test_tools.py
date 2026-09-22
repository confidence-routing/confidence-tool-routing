"""
test_tools.py
=============

Unit tests for eval_harness.tools -- the calculator and the Python runner.

The calculator tests carry more weight than their size suggests: it is a
trust boundary, model output flowing into an evaluator. The refusal cases
below are the ones that matter, and they are written as assertions rather
than left to a code review of the whitelist.

Run:
    python -m pytest tests/ -v
    (or, without pytest installed: python -m tests.test_tools)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness.tools import calculate, run_python


# ---------------------------------------------------------------------------
# calculate -- arithmetic it should do
# ---------------------------------------------------------------------------

def test_respects_operator_precedence():
    # 2 + (3*4) = 14, not (2+3)*4 = 20
    assert calculate("2+3*4") == "14"
    print("test_respects_operator_precedence: PASS")


def test_strips_currency_and_thousands_separators():
    # models write money as "$1,200"; 1200/4 = 300
    assert calculate("$1,200/4") == "300"
    print("test_strips_currency_and_thousands_separators: PASS")


def test_integral_division_renders_without_decimal_point():
    # 48/2 is 24.0 as a float; it must render "24" so a string compare
    # against a gold answer of "24" does not fail on formatting alone.
    assert calculate("48/2") == "24"
    print("test_integral_division_renders_without_decimal_point: PASS")


def test_non_integral_division_keeps_precision():
    # 10/3 rounded to 10dp
    assert calculate("10/3") == "3.3333333333"
    print("test_non_integral_division_keeps_precision: PASS")


def test_handles_negative_and_parenthesised():
    assert calculate("-(3+4)*2") == "-14"
    print("test_handles_negative_and_parenthesised: PASS")


def test_trailing_equals_is_tolerated():
    # models emit "12*12=" when asked for an expression
    assert calculate("12*12=") == "144"
    print("test_trailing_equals_is_tolerated: PASS")


# ---------------------------------------------------------------------------
# calculate -- input it must refuse
#
# These are the trust-boundary cases. calculate() never calls eval(); it
# parses to an AST and checks every node against a whitelist, so anything
# that is not arithmetic returns None instead of executing.
# ---------------------------------------------------------------------------

def test_refuses_code_execution():
    # the case the whitelist exists for
    assert calculate('__import__("os").system("echo pwned")') is None
    print("test_refuses_code_execution: PASS")


def test_refuses_function_calls_and_names():
    assert calculate("len([1,2,3])") is None
    assert calculate("open('/etc/passwd').read()") is None
    assert calculate("x + 1") is None
    print("test_refuses_function_calls_and_names: PASS")


def test_refuses_exponent_bomb():
    # 9**9**9 would materialise a number with billions of digits and hang
    # the process long before any timeout in the caller fires.
    assert calculate("9**9**9") is None
    print("test_refuses_exponent_bomb: PASS")


def test_division_by_zero_is_none_not_a_crash():
    # the caller is a routing loop: a tool that cannot answer returns "no
    # answer", it does not take the run down.
    assert calculate("1/0") is None
    print("test_division_by_zero_is_none_not_a_crash: PASS")


def test_empty_and_non_string_input():
    assert calculate("") is None
    assert calculate("   ") is None
    assert calculate(None) is None
    assert calculate("$") is None
    print("test_empty_and_non_string_input: PASS")


def test_booleans_are_not_numbers():
    # bool is an int subclass in Python; "True+True" must not be 2.
    assert calculate("True+True") is None
    print("test_booleans_are_not_numbers: PASS")


# ---------------------------------------------------------------------------
# run_python
# ---------------------------------------------------------------------------

def test_runs_and_captures_stdout():
    result = run_python("print(6*7)")
    assert result.ok is True, result.stderr
    assert result.stdout.strip() == "42"
    print("test_runs_and_captures_stdout: PASS")


def test_failing_program_reports_not_ok():
    result = run_python("raise ValueError('nope')")
    assert result.ok is False
    assert "ValueError" in result.stderr
    assert result.timed_out is False
    print("test_failing_program_reports_not_ok: PASS")


def test_infinite_loop_times_out():
    # the reason the runner takes a timeout at all: generated code hangs.
    result = run_python("while True: pass", timeout=1.0)
    assert result.ok is False
    assert result.timed_out is True
    print("test_infinite_loop_times_out: PASS")


def test_empty_program_is_rejected():
    assert run_python("").ok is False
    assert run_python("   ").ok is False
    print("test_empty_program_is_rejected: PASS")


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
