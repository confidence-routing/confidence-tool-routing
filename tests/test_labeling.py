"""
test_labeling.py
================

Unit tests for eval_harness.labeling, driven by a stub client.

Run:
    python -m pytest tests/ -v
    (or, without pytest installed: python -m tests.test_labeling)
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness.labeling import (
    apply_labels, decide_label, label_tool_necessity, load_labels,
    save_labels, split_tasks, summarize,
)

GSM = {"task_id": "g1", "dataset": "gsm8k", "query": "6*7?", "gold_answer": "42"}


def _resp(text):
    return {"choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2}}


class ScriptedClient:
    """Returns a fixed sequence of answers per sample() call."""
    def __init__(self, answers):
        self.answers = answers
    def sample(self, messages, model, k=3, **kw):
        return [_resp(a) for a in self.answers[:k]]


def test_always_right_means_no_tool_was_needed():
    out = label_tool_necessity(ScriptedClient(["42", "42", "42"]), [GSM], "m", trials=3)
    assert out["labels"]["g1"] == "not_required"
    assert out["n_correct"]["g1"] == 3
    print("test_always_right_means_no_tool_was_needed: PASS")


def test_never_right_means_a_tool_was_needed():
    out = label_tool_necessity(ScriptedClient(["41", "43", "40"]), [GSM], "m", trials=3)
    assert out["labels"]["g1"] == "required"
    print("test_never_right_means_a_tool_was_needed: PASS")


def test_inconsistent_is_ambiguous_not_a_coin_flip():
    # Sometimes right is genuinely a third state. Forcing it into either
    # bucket would put a task the model half-knows into ground truth as
    # if it were certain, and AMBIGUOUS is already excluded from routing
    # metrics by default.
    out = label_tool_necessity(ScriptedClient(["42", "41", "42"]), [GSM], "m", trials=3)
    assert out["labels"]["g1"] == "ambiguous"
    print("test_inconsistent_is_ambiguous_not_a_coin_flip: PASS")


def test_one_trial_cannot_distinguish_reliable_from_lucky():
    # Documented consequence of trials=1: every task lands on a hard
    # label, because there is no "sometimes" to observe.
    out = label_tool_necessity(ScriptedClient(["42"]), [GSM], "m", trials=1)
    assert out["labels"]["g1"] == "not_required"
    assert decide_label(0, 1) == "required"
    print("test_one_trial_cannot_distinguish_reliable_from_lucky: PASS")


def test_label_set_records_the_model_it_belongs_to():
    # A label is model-relative: a task one model answers from memory may
    # genuinely need a calculator for a smaller one.
    out = label_tool_necessity(ScriptedClient(["42"]*3), [GSM], "gpt-oss-120b", trials=3)
    assert out["model"] == "gpt-oss-120b"
    assert out["trials"] == 3
    print("test_label_set_records_the_model_it_belongs_to: PASS")


def test_apply_labels_attaches_without_mutating_the_input():
    labels = {"labels": {"g1": "required"}}
    out = apply_labels([GSM], labels)
    assert out[0]["tool_necessity"] == "required"
    assert "tool_necessity" not in GSM, "input task must not be mutated"
    print("test_apply_labels_attaches_without_mutating_the_input: PASS")


def test_unlabelled_tasks_are_dropped_by_default():
    # Letting them through unlabelled would silently assert they needed
    # no tool -- inventing the ground truth the pilot exists to measure.
    out = apply_labels([GSM, dict(GSM, task_id="g2")], {"labels": {"g1": "required"}})
    assert [t["task_id"] for t in out] == ["g1"]
    lenient = apply_labels([GSM, dict(GSM, task_id="g2")],
                           {"labels": {"g1": "required"}}, strict=False)
    assert len(lenient) == 2
    print("test_unlabelled_tasks_are_dropped_by_default: PASS")


def test_labels_round_trip_through_disk():
    out = label_tool_necessity(ScriptedClient(["42"]*3), [GSM], "m", trials=3)
    path = Path(tempfile.mkdtemp()) / "labels.json"
    save_labels(out, str(path))
    assert load_labels(str(path))["labels"] == out["labels"]
    print("test_labels_round_trip_through_disk: PASS")


def test_pilot_and_eval_splits_are_disjoint_and_deterministic():
    # The router must not be scored on the tasks whose labels it was
    # tuned against.
    tasks = [{"task_id": str(i)} for i in range(20)]
    a = split_tasks(tasks, pilot_fraction=0.5, seed=1)
    b = split_tasks(tasks, pilot_fraction=0.5, seed=1)
    pilot_ids = {t["task_id"] for t in a["pilot"]}
    eval_ids = {t["task_id"] for t in a["eval"]}
    assert not (pilot_ids & eval_ids)
    assert pilot_ids | eval_ids == {str(i) for i in range(20)}
    assert [t["task_id"] for t in a["pilot"]] == [t["task_id"] for t in b["pilot"]]
    print("test_pilot_and_eval_splits_are_disjoint_and_deterministic: PASS")


def test_summarize_counts_every_label_kind():
    counts = summarize({"labels": {"a": "required", "b": "required",
                                   "c": "not_required"}})
    assert counts == {"required": 2, "not_required": 1, "ambiguous": 0}
    print("test_summarize_counts_every_label_kind: PASS")


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
