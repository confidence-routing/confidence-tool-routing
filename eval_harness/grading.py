"""
grading.py
==========

Fills in ``TaskRecord.correct``. One grader per dataset, chosen because
both grade themselves without a human or a judge model in the loop:

    gsm8k      final number matches the gold number
    humaneval  the code passes the test suite the dataset ships with

Automatic grading is why these two datasets were picked for the shipping
scope. A grader that needs an LLM to judge correctness would put a second
unvalidated model inside the metric the whole experiment reports, which
is a problem you cannot argue your way out of in a results table.

Answer normalization is NOT reimplemented here. ``confidence.py`` already
canonicalizes free-text answers for vote bucketing -- case, whitespace,
answer prefixes, "$1,234.00" == "1234" -- and grading needs the same
rules. Two different notions of "same answer" in one codebase is how you
end up with a self-consistency score that disagrees with the correctness
label on the same pair of strings. The one thing added here is pulling a
final number out of a sentence, which vote bucketing never needed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .confidence import normalize_answer
from .tools import run_python

# A number, possibly signed, comma-grouped, decimal, or currency-prefixed.
_NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*\.?\d*")


def extract_final_number(text: Any) -> Optional[str]:
    """
    The last number in a string, canonicalized.

    Last rather than first because a worked answer states its result at
    the end ("... so he saves 12 dollars a week, for a total of 624").
    Taking the first number would grade the model on an intermediate step
    -- and GSM8K answers are chains of intermediate steps, so this is the
    common case, not an edge one.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)

    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None

    # Walk backwards: the last match may be a bare "." or "-" artifact
    # that normalize_answer rejects, in which case the real answer is the
    # one before it.
    for raw in reversed(matches):
        canonical = normalize_answer(raw.rstrip("."))
        if canonical is not None:
            return canonical
    return None


def grade_gsm8k(model_answer: Any, gold_answer: Any) -> bool:
    """True if the model's final number matches the gold number."""
    predicted = extract_final_number(model_answer)
    gold = extract_final_number(gold_answer)
    if predicted is None or gold is None:
        return False
    return predicted == gold


def _strip_code_fences(text: str) -> str:
    """Drop a markdown code fence if the model wrapped its answer in one."""
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return fenced.group(1) if fenced else text


def build_humaneval_program(completion: str, meta: Dict[str, Any]) -> Optional[str]:
    """
    Assemble the program that gets executed: candidate code, then the
    dataset's own tests, then the call that runs them.

    Models answer this task two ways -- some restate the whole function,
    some return only the body to be appended to the prompt. Guessing
    wrong produces either a duplicate definition or an unindented
    fragment, and both look like a wrong answer rather than a harness
    bug, so check which one arrived instead of assuming.
    """
    prompt = meta.get("prompt", "")
    test_code = meta.get("test_code", "")
    entry_point = meta.get("entry_point", "")
    if not test_code or not entry_point:
        return None

    code = _strip_code_fences(completion or "").strip("\n")
    if not code.strip():
        return None

    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, re.MULTILINE):
        candidate = code           # a complete function: use it as-is
    else:
        # A bare body belongs after the signature -- with a newline forced
        # in between. The prompt ends on the docstring's closing quotes,
        # and any loader that .strip()s it drops that final newline, which
        # glues the first line of the body onto the """ and turns every
        # task in the dataset into a SyntaxError. Cheap to defend against
        # here, and invisible when it goes wrong: it reads as a model that
        # cannot write Python.
        candidate = prompt.rstrip("\n") + "\n" + code

    return f"{candidate}\n\n{test_code}\n\ncheck({entry_point})\n"


def grade_humaneval(completion: str, meta: Dict[str, Any], timeout: float = 10.0) -> bool:
    """True if the candidate code passes every test the dataset ships."""
    program = build_humaneval_program(completion, meta)
    if program is None:
        return False
    return run_python(program, timeout=timeout).ok


def grade(dataset: str, model_answer: Any, gold_answer: Any = None,
          meta: Optional[Dict[str, Any]] = None) -> bool:
    """Dispatch to the right grader. Unknown dataset is a hard error --
    silently grading everything False would read as a terrible model."""
    if dataset == "gsm8k":
        return grade_gsm8k(model_answer, gold_answer)
    if dataset == "humaneval":
        return grade_humaneval(model_answer, meta or {})
    raise KeyError(f"No grader for dataset '{dataset}'. Add one in grading.py.")
