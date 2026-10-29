"""
grading.py
==========

Fills in ``TaskRecord.correct``. One grader per dataset, chosen because
both grade themselves without a human or a judge model in the loop:

    gsm8k        final number matches the gold number
    humaneval    the code passes the test suite the dataset ships with
    coqa         token-overlap F1 against the reference answer
    overruling   binary Yes/No verdict
    headlines    category label match

Automatic grading is why these two datasets were picked for the shipping
scope. A grader that needs an LLM to judge correctness would put a second
unvalidated model inside the metric the whole experiment reports, which
is a problem you cannot argue your way out of in a results table.

CoQA is the one place confidence.py's normalization is deliberately NOT
reused. Its published numbers are SQuAD-style token F1, which strips
articles and punctuation before comparing token bags -- a convention, not
a better idea. Deviating from it would make the results incomparable to
every other CoQA number in the literature, which is worse than having two
normalizers. The yes/no and category graders below do reuse what exists.

Answer normalization is otherwise NOT reimplemented here. ``confidence.py`` already
canonicalizes free-text answers for vote bucketing -- case, whitespace,
answer prefixes, "$1,234.00" == "1234" -- and grading needs the same
rules. Two different notions of "same answer" in one codebase is how you
end up with a self-consistency score that disagrees with the correctness
label on the same pair of strings. The one thing added here is pulling a
final number out of a sentence, which vote bucketing never needed.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from .confidence import _verdict_key, normalize_answer
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


# ---------------------------------------------------------------------------
# CoQA -- token-overlap F1
# ---------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}


def _normalize_qa(text: Any) -> List[str]:
    """
    SQuAD/CoQA normalization, in order: casefold, DELETE punctuation, drop
    articles, split on whitespace.

    Deleting punctuation rather than replacing it with a space is the
    detail that matters, and it is the published convention:
    "down-stream" has to become one token, "downstream", or it scores 0
    against it -- the model marked wrong purely for hyphenating.
    """
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
    stripped = "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace())
    return [t for t in stripped.split() if t not in _ARTICLES]


def token_f1(prediction: Any, reference: Any) -> float:
    """
    Token-overlap F1 in [0, 1], the standard CoQA metric.

    Free-text answers do not admit exact match: "down-stream", "he went
    down-stream" and "downstream" are one answer, and grading them as
    three understates every model equally and means nothing.

    Yes/no answers are the known weak spot -- one token makes F1
    all-or-nothing there -- which is a property of the metric rather than
    this implementation, and what published CoQA numbers use.
    """
    pred, ref = _normalize_qa(prediction), _normalize_qa(reference)
    if not pred or not ref:
        return float(pred == ref)     # both empty is a match; one empty is not

    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


# TaskRecord.correct is a bool, so the continuous F1 has to be cut
# somewhere. 0.5 is the conventional "substantially right" line, and the
# runner keeps the raw score in meta so a different cut can be applied
# afterwards without paying for the run again.
COQA_F1_THRESHOLD = 0.5


def grade_coqa(model_answer: Any, gold_answer: Any,
               threshold: float = COQA_F1_THRESHOLD) -> bool:
    """True if token-overlap F1 clears ``threshold``."""
    return token_f1(model_answer, gold_answer) >= threshold


# ---------------------------------------------------------------------------
# Overruling -- binary Yes/No
# ---------------------------------------------------------------------------

def extract_verdict(text: Any) -> Optional[str]:
    """
    "yes" or "no" from a reply, or None if it committed to neither.

    Scans word by word and takes the FIRST verdict token, because a model
    asked a yes/no question answers it and then justifies: "Yes. The
    sentence explicitly rejects the earlier holding" also contains
    "rejects". Taking the last match would grade the justification.

    None rather than a default: a reply that never says yes or no is a
    non-answer, and scoring it as "no" would credit the model for every
    negative case it dodged. The yes/no vocabulary is confidence.py's,
    not a second list.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    for raw in text.split():
        verdict = _verdict_key(raw)
        if verdict is not None:
            return verdict
    return None


def grade_binary(model_answer: Any, gold_answer: Any) -> bool:
    """True if the reply's verdict matches the gold Yes/No label."""
    predicted = extract_verdict(model_answer)
    gold = extract_verdict(gold_answer)
    if predicted is None or gold is None:
        return False
    return predicted == gold


# ---------------------------------------------------------------------------
# Headlines -- category label
# ---------------------------------------------------------------------------

def _normalize_category(text: Any) -> str:
    """Casefold, delete punctuation, collapse whitespace. "U.S. NEWS" and
    "us news" are the same label; punctuation is formatting."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return " ".join("".join(
        ch for ch in text.lower() if ch.isalnum() or ch.isspace()).split())


def grade_category(model_answer: Any, gold_answer: Any) -> bool:
    """
    True if the reply names the gold category.

    Accepts the label alone or wrapped in a sentence ("The category is
    U.S. NEWS"), since a classifier prompt does not reliably suppress
    prose. Matching is on the whole normalized label as a phrase, not on
    individual words: several categories share a word -- U.S. NEWS, WORLD
    NEWS, WEIRD NEWS -- and word-level matching would count any of them
    as any other.
    """
    gold = _normalize_category(gold_answer)
    if not gold:
        return False
    predicted = _normalize_category(model_answer)
    if not predicted:
        return False
    return predicted == gold or f" {gold} " in f" {predicted} "


def grade(dataset: str, model_answer: Any, gold_answer: Any = None,
          meta: Optional[Dict[str, Any]] = None) -> bool:
    """Dispatch to the right grader. Unknown dataset is a hard error --
    silently grading everything False would read as a terrible model."""
    if dataset == "gsm8k":
        return grade_gsm8k(model_answer, gold_answer)
    if dataset == "humaneval":
        return grade_humaneval(model_answer, meta or {})
    if dataset == "coqa":
        return grade_coqa(model_answer, gold_answer)
    if dataset == "overruling":
        return grade_binary(model_answer, gold_answer)
    if dataset == "headlines":
        return grade_category(model_answer, gold_answer)
    raise KeyError(f"No grader for dataset '{dataset}'. Add one in grading.py.")
