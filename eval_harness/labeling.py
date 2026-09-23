"""
labeling.py
===========

Produces ``tool_necessity``: did this task actually need a tool?

Every routing metric grades against this label -- precision, recall,
unnecessary-call rate, missed-call rate all key off it -- and nothing
else in the harness produces it. Loaders deliberately do not: which tool
a task would use is a property of the dataset, whether it needs one is
not.

How the label is decided
------------------------
Empirically, by asking the model the question with no tool, several
times, and grading each attempt:

    always right   -> not_required   (it never needed help)
    never right    -> required       (it could not do this alone)
    sometimes      -> ambiguous      (excluded from routing metrics by
                                      default, which is what AMBIGUOUS is
                                      for)

Sampling more than once is the point. A single attempt cannot tell a
task the model reliably solves from one it guessed, and those are
different labels.

Two things to be honest about
-----------------------------
The label is MODEL-RELATIVE. A task gpt-oss-120b answers from memory may
genuinely need a calculator for a smaller model. That is a real property
of the question "did this need a tool", not a flaw -- but it means a
label set belongs to the model that produced it, which is why
save_labels records the model and the trial count alongside.

And it is circular if you are careless. Defining "needed a tool" as "the
model failed without one" and then grading the router on the same tasks
measures the router against the model's own failures. The standard
guard, and the one the pilot script applies, is to label a held-out
split and evaluate on a different one, so the router is never scored on
the tasks whose labels it was tuned against. Say so in the writeup
either way -- it is a definition, not a measurement.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .grading import grade
from .models import ToolNecessity

DEFAULT_TRIALS = 3
DEFAULT_TEMPERATURE = 0.8


def decide_label(n_correct: int, trials: int) -> str:
    """All right -> not_required, none right -> required, else ambiguous."""
    if trials <= 0:
        raise ValueError(f"trials must be >= 1, got {trials}")
    if n_correct >= trials:
        return ToolNecessity.NOT_REQUIRED.value
    if n_correct <= 0:
        return ToolNecessity.REQUIRED.value
    return ToolNecessity.AMBIGUOUS.value


def label_tool_necessity(
    client: Any,
    tasks: Sequence[Dict[str, Any]],
    model: str,
    *,
    trials: int = DEFAULT_TRIALS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = 512,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Label each task by attempting it ``trials`` times with no tool.

    Returns a dict ready for save_labels(), carrying the model and trial
    count as well as the labels -- a label set is only meaningful next to
    the model that produced it.
    """
    from .runner import build_messages   # local: avoids an import cycle

    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")

    labels: Dict[str, str] = {}
    detail: Dict[str, int] = {}

    for i, task in enumerate(tasks):
        task_id = str(task.get("task_id", i))
        # build_messages with no passage IS the no-tool condition: for
        # CoQA the conversation history still goes in, since without it
        # the question is unanswerable for reasons that have nothing to
        # do with retrieval.
        messages = build_messages(task)
        samples = client.sample(messages, model, k=trials,
                                temperature=temperature, max_tokens=max_tokens)

        n_correct = 0
        for sample in samples:
            answer = _answer_text(sample)
            try:
                if grade(task.get("dataset", ""), answer,
                         gold_answer=task.get("gold_answer"),
                         meta=task.get("meta")):
                    n_correct += 1
            except KeyError:
                raise   # an unknown dataset is a missing grader, not a hard task

        label = decide_label(n_correct, trials)
        labels[task_id] = label
        detail[task_id] = n_correct
        if progress:
            progress(i + 1, len(tasks), label)

    return {
        "model": model,
        "trials": trials,
        "temperature": temperature,
        "created": time.time(),
        "labels": labels,
        "n_correct": detail,
    }


def _answer_text(response: Any) -> str:
    from .confidence import _get
    choices = _get(response, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        return ""
    return (_get(_get(choices[0], "message"), "content") or "").strip()


def summarize(label_set: Dict[str, Any]) -> Dict[str, int]:
    """Counts per label, for checking a split is usable before spending on it."""
    counts = {t.value: 0 for t in ToolNecessity}
    for label in label_set.get("labels", {}).values():
        counts[label] = counts.get(label, 0) + 1
    return counts


def save_labels(label_set: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(label_set, fh, indent=2, sort_keys=True)


def load_labels(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def apply_labels(
    tasks: Sequence[Dict[str, Any]], label_set: Dict[str, Any], *, strict: bool = True
) -> List[Dict[str, Any]]:
    """
    Attach labels to tasks, returning copies.

    strict=True drops tasks with no label rather than letting them
    through unlabelled. Unset reads as NOT_REQUIRED downstream, so a
    silent pass-through would quietly assert that every unlabelled task
    needed no tool -- inventing exactly the ground truth the pilot exists
    to measure.
    """
    labels = label_set.get("labels", {})
    out: List[Dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("task_id", ""))
        label = labels.get(task_id)
        if label is None:
            if strict:
                continue
            out.append(dict(task))
            continue
        out.append(dict(task, tool_necessity=label))
    return out


def split_tasks(
    tasks: Sequence[Dict[str, Any]], *, pilot_fraction: float = 0.5, seed: int = 0
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Deterministic pilot/eval split.

    The router must not be scored on the tasks whose labels it was tuned
    against, or "needed a tool" and "the router escalated" are two
    readings of the same measurement.
    """
    if not 0.0 < pilot_fraction < 1.0:
        raise ValueError(f"pilot_fraction must be in (0, 1), got {pilot_fraction}")
    import random as _random

    shuffled = list(tasks)
    _random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * pilot_fraction))
    return {"pilot": shuffled[:cut], "eval": shuffled[cut:]}
