"""
test_grading.py
===============

Unit tests for eval_harness.grading.

The HumanEval fixture below is a miniature of the real thing -- a prompt
ending on a docstring, a bare-body completion, and a check() test -- so
these run without downloading the dataset. The full dataset is the other
half of the verification: all 164 canonical solutions pass their own
tests, and all 1319 GSM8K gold answers grade correct against themselves.

Run:
    python -m pytest tests/ -v
    (or, without pytest installed: python -m tests.test_grading)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness.grading import (
    build_humaneval_program,
    grade_coqa,
    token_f1,
    extract_final_number,
    grade,
    grade_gsm8k,
    grade_humaneval,
)

# Mirrors HumanEval's shape: the prompt ends on the docstring's closing
# quotes AND A TRAILING NEWLINE, and the completion is a bare body.
_PROMPT = (
    'def add_one(n: int) -> int:\n'
    '    """Return n + 1.\n'
    '\n'
    '    >>> add_one(1)\n'
    '    2\n'
    '    """\n'
)
_TEST = (
    'def check(candidate):\n'
    '    assert candidate(1) == 2\n'
    '    assert candidate(-1) == 0\n'
)
_META = {"prompt": _PROMPT, "test_code": _TEST, "entry_point": "add_one"}
_BODY = "    return n + 1\n"


# ---------------------------------------------------------------------------
# extract_final_number
# ---------------------------------------------------------------------------

def test_takes_the_last_number_not_the_first():
    # The one that matters. A worked GSM8K answer is a chain of
    # intermediate steps and states its result at the end; taking the
    # first number grades the model on step one.
    text = "He buys 3 packs at 12 each, so 36, then spends 4 more: 40"
    assert extract_final_number(text) == "40"
    print("test_takes_the_last_number_not_the_first: PASS")


def test_ignores_sentence_final_period():
    # "52." must not parse as the float 52. and render "52.0"
    assert extract_final_number("So Jared types 52 words per minute.") == "52"
    print("test_ignores_sentence_final_period: PASS")


def test_canonicalises_currency_and_separators():
    assert extract_final_number("The answer is $1,234.00") == "1234"
    print("test_canonicalises_currency_and_separators: PASS")


def test_no_number_returns_none():
    assert extract_final_number("I do not know") is None
    assert extract_final_number("") is None
    assert extract_final_number(None) is None
    print("test_no_number_returns_none: PASS")


# ---------------------------------------------------------------------------
# grade_gsm8k
# ---------------------------------------------------------------------------

def test_gsm8k_verbose_answer_matches_bare_gold():
    assert grade_gsm8k("So Jared types 52 words per minute.", "52") is True
    print("test_gsm8k_verbose_answer_matches_bare_gold: PASS")


def test_gsm8k_formatting_differences_still_match():
    # "$1,234.00" and "1234" are the same answer
    assert grade_gsm8k("The answer is $1,234.00", "1234") is True
    print("test_gsm8k_formatting_differences_still_match: PASS")


def test_gsm8k_wrong_number_fails():
    assert grade_gsm8k("I think it is 41", "52") is False
    print("test_gsm8k_wrong_number_fails: PASS")


def test_gsm8k_unanswerable_is_false_not_an_error():
    assert grade_gsm8k("I do not know", "52") is False
    assert grade_gsm8k(None, "52") is False
    print("test_gsm8k_unanswerable_is_false_not_an_error: PASS")


# ---------------------------------------------------------------------------
# HumanEval program assembly
# ---------------------------------------------------------------------------

def test_bare_body_is_appended_to_the_prompt():
    program = build_humaneval_program(_BODY, _META)
    assert "def add_one" in program
    assert "return n + 1" in program
    assert program.rstrip().endswith("check(add_one)")
    assert grade_humaneval(_BODY, _META) is True
    print("test_bare_body_is_appended_to_the_prompt: PASS")


def test_stripped_prompt_still_assembles():
    # REGRESSION. The dataset loader .strip()s the prompt, which removes
    # the newline after the docstring's closing quotes. Concatenating a
    # bare body onto that produces:
    #     """    return n + 1
    # which is a SyntaxError, and every task in the dataset fails. It
    # reads as a model that cannot write Python, not as a harness bug --
    # measured on the real data, 4/164 before the newline was forced back
    # and 164/164 after.
    stripped = dict(_META, prompt=_PROMPT.strip())
    program = build_humaneval_program(_BODY, stripped)
    assert '"""    return' not in program, program
    assert grade_humaneval(_BODY, stripped) is True
    print("test_stripped_prompt_still_assembles: PASS")


def test_full_function_completion_is_used_as_is():
    # Some models restate the whole function instead of continuing it.
    # Appending that to the prompt would define add_one twice.
    full = 'def add_one(n: int) -> int:\n    return n + 1\n'
    program = build_humaneval_program(full, _META)
    assert program.count("def add_one") == 1, program
    assert grade_humaneval(full, _META) is True
    print("test_full_function_completion_is_used_as_is: PASS")


def test_markdown_fences_are_stripped():
    fenced = "```python\ndef add_one(n: int) -> int:\n    return n + 1\n```"
    assert grade_humaneval(fenced, _META) is True
    print("test_markdown_fences_are_stripped: PASS")


def test_wrong_implementation_fails_the_tests():
    assert grade_humaneval("    return n + 2\n", _META) is False
    print("test_wrong_implementation_fails_the_tests: PASS")


def test_empty_or_unusable_completion_is_false():
    assert grade_humaneval("", _META) is False
    assert build_humaneval_program(_BODY, {"prompt": _PROMPT}) is None
    print("test_empty_or_unusable_completion_is_false: PASS")


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def test_grade_dispatches_by_dataset():
    assert grade("gsm8k", "the answer is 52", "52") is True
    assert grade("humaneval", _BODY, meta=_META) is True
    print("test_grade_dispatches_by_dataset: PASS")


def test_unknown_dataset_raises():
    # Silently returning False would read as a terrible model rather than
    # a missing grader.
    try:
        grade("not_a_dataset", "x", "y")
    except KeyError:
        print("test_unknown_dataset_raises: PASS")
        return
    raise AssertionError("expected KeyError for an unknown dataset")


# ---------------------------------------------------------------------------
# CoQA -- token-overlap F1
# ---------------------------------------------------------------------------

def test_identical_answers_score_one():
    assert token_f1("down-stream", "down-stream") == 1.0
    print("test_identical_answers_score_one: PASS")


def test_punctuation_is_deleted_not_spaced():
    # The detail that makes this match published CoQA numbers. Deleting
    # the hyphen gives one token, "downstream", which matches. Replacing
    # it with a space gives two tokens and zero overlap -- the model
    # marked wrong purely for hyphenating.
    assert token_f1("downstream", "down-stream") == 1.0
    print("test_punctuation_is_deleted_not_spaced: PASS")


def test_articles_are_ignored():
    assert token_f1("the dog", "a dog") == 1.0
    print("test_articles_are_ignored: PASS")


def test_partial_overlap_scores_between():
    # pred={he,went,downstream} ref={downstream}
    #   precision 1/3, recall 1/1 -> f1 = 2*(1/3)/(1/3+1) = 0.5
    assert abs(token_f1("he went down-stream", "down-stream") - 0.5) < 1e-9
    print("test_partial_overlap_scores_between: PASS")


def test_no_overlap_scores_zero():
    assert token_f1("upstream", "down-stream") == 0.0
    print("test_no_overlap_scores_zero: PASS")


def test_empty_prediction_scores_zero_against_a_real_answer():
    assert token_f1("", "down-stream") == 0.0
    assert token_f1(None, "down-stream") == 0.0
    # both empty is a match, not a failure
    assert token_f1("", "") == 1.0
    print("test_empty_prediction_scores_zero_against_a_real_answer: PASS")


def test_grade_coqa_cuts_at_the_threshold():
    # 0.5 exactly must pass: the cut is >=, and half-overlap answers like
    # "he went down-stream" land precisely there rather than near it.
    assert grade_coqa("he went down-stream", "down-stream") is True
    assert grade_coqa("upstream", "down-stream") is False
    assert grade_coqa("he went somewhere down-stream today", "down-stream") is False
    print("test_grade_coqa_cuts_at_the_threshold: PASS")


def test_grade_dispatches_coqa():
    assert grade("coqa", "Joe Ladue.", "joe ladue") is True
    print("test_grade_dispatches_coqa: PASS")


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
