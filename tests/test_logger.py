"""
test_logger.py
==============

Unit tests for eval_harness.logger.

Mostly about one failure mode: append mode is deliberate (a crashed run
keeps its rows and can be continued), but it means re-running a config
under a run_id that already has rows silently doubles the data, and
every metric is then computed across two runs at once. The numbers stay
plausible, which is exactly what makes it worth a guard.

Run:
    python -m pytest tests/ -v
    (or, without pytest installed: python -m tests.test_logger)
"""

import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness.logger import RunLogger
from eval_harness.models import TaskRecord


def _write(run_id, log_dir, n=3, **kwargs):
    with RunLogger(run_id, log_dir=log_dir, **kwargs) as lg:
        for i in range(n):
            lg.log(TaskRecord(task_id=str(i), dataset="d"))


def test_reusing_a_run_id_warns():
    d = tempfile.mkdtemp()
    _write("exp", d)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _write("exp", d)
    assert any("already has rows" in str(w.message) for w in caught), \
        [str(w.message) for w in caught]
    print("test_reusing_a_run_id_warns: PASS")


def test_the_warning_is_the_only_thing_that_changes():
    # The guard must not delete or truncate: a half-finished run is data,
    # and the caller decides what to do with it.
    d = tempfile.mkdtemp()
    _write("exp", d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _write("exp", d)
    assert len(RunLogger.load(f"{d}/exp.jsonl")) == 6
    print("test_the_warning_is_the_only_thing_that_changes: PASS")


def test_resume_true_is_silent():
    # Continuing a crashed run is the case append mode exists for.
    d = tempfile.mkdtemp()
    _write("exp", d)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _write("exp", d, resume=True)
    assert not any("already has rows" in str(w.message) for w in caught)
    print("test_resume_true_is_silent: PASS")


def test_a_fresh_run_id_is_silent():
    d = tempfile.mkdtemp()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _write("brand_new", d)
    assert not any("already has rows" in str(w.message) for w in caught)
    print("test_a_fresh_run_id_is_silent: PASS")


def test_an_empty_log_file_is_not_a_reuse():
    # Touched but never written -- an interrupted start, not a prior run.
    d = tempfile.mkdtemp()
    Path(d, "exp.jsonl").write_text("", encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _write("exp", d)
    assert not any("already has rows" in str(w.message) for w in caught)
    print("test_an_empty_log_file_is_not_a_reuse: PASS")


def test_round_trip_preserves_records():
    d = tempfile.mkdtemp()
    with RunLogger("rt", log_dir=d) as lg:
        lg.log(TaskRecord(task_id="a", dataset="gsm8k", confidence_score=0.5))
    loaded = RunLogger.load(f"{d}/rt.jsonl")
    assert len(loaded) == 1
    assert loaded[0].task_id == "a"
    assert loaded[0].confidence_score == 0.5
    assert loaded[0].run_id == "rt", "the logger stamps its run_id onto rows"
    print("test_round_trip_preserves_records: PASS")


def test_iter_load_matches_load():
    d = tempfile.mkdtemp()
    _write("it", d, n=4)
    assert [r.task_id for r in RunLogger.iter_load(f"{d}/it.jsonl")] == \
           [r.task_id for r in RunLogger.load(f"{d}/it.jsonl")]
    print("test_iter_load_matches_load: PASS")


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
