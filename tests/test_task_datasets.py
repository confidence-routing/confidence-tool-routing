"""
test_task_datasets.py
=====================

Unit tests for the dataset loaders.

Everything here runs offline. The network-dependent half (download) is
not mocked -- it is simply not exercised: each test writes a small
fixture into a temporary data_dir and calls parse() directly, which is
where all the logic that can actually be wrong lives.

Run:
    python -m pytest tests/ -v
    (or, without pytest installed: python -m tests.test_task_datasets)
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from task_datasets.base import DatasetLoader, TaskItem, load_dataset, list_datasets
from task_datasets.gsm8k import GSM8KLoader, _extract_final_answer
from task_datasets.headlines import _cached_rows
from task_datasets.humaneval import HumanEvalLoader


def _tmp_loader(cls, **files):
    """A loader pointed at a throwaway data_dir seeded with fixture files."""
    tmp = Path(tempfile.mkdtemp())
    loader = cls(data_dir=tmp)
    for name, content in files.items():
        (tmp / name.replace("__", ".")).write_text(content, encoding="utf-8")
    return loader


# ---------------------------------------------------------------------------
# GSM8K answer extraction
# ---------------------------------------------------------------------------

def test_extract_final_answer_reads_the_hash_marker():
    # GSM8K answers end "#### <number>" after the worked solution
    assert _extract_final_answer("Some working.\n#### 52") == "52"
    print("test_extract_final_answer_reads_the_hash_marker: PASS")


def test_extract_final_answer_strips_thousands_separators():
    assert _extract_final_answer("working\n#### 1,234") == "1234"
    print("test_extract_final_answer_strips_thousands_separators: PASS")


def test_extract_final_answer_without_marker_falls_back():
    # Rather than returning None and silently producing a task with no
    # gold answer, which would grade as wrong for every model.
    assert _extract_final_answer("just 42") == "just 42"
    print("test_extract_final_answer_without_marker_falls_back: PASS")


def test_gsm8k_parse_builds_task_items():
    rows = "\n".join(json.dumps(r) for r in [
        {"question": "What is 6*7?", "answer": "reasoning\n#### 42"},
        {"question": "", "answer": "#### 1"},          # blank -> skipped
    ])
    loader = _tmp_loader(GSM8KLoader, test__jsonl=rows)
    items = loader.parse()
    assert len(items) == 1, items
    assert items[0]["gold_answer"] == "42"
    assert items[0]["tool_type"] == "calculator"
    assert items[0]["dataset"] == "gsm8k"
    print("test_gsm8k_parse_builds_task_items: PASS")


# ---------------------------------------------------------------------------
# HumanEval prompt handling -- regression
# ---------------------------------------------------------------------------

_PROMPT = 'def add_one(n):\n    """Return n + 1.\n    """\n'


def test_humaneval_prompt_keeps_its_trailing_newline():
    # REGRESSION. The prompt used to be .strip()ed, which removed the
    # newline after the docstring's closing quotes. Completions are bare
    # function bodies appended straight on, so stripping it fused the
    # first line of the body onto the """ and made every one of the 164
    # tasks a SyntaxError -- 4/164 graded correct instead of 164/164.
    row = json.dumps({"task_id": "HumanEval/0", "prompt": _PROMPT,
                      "canonical_solution": "    return n + 1\n",
                      "test": "def check(c):\n    assert c(1)==2\n",
                      "entry_point": "add_one"})
    loader = _tmp_loader(HumanEvalLoader, HumanEval__jsonl=row)
    item = loader.parse()[0]
    assert item["meta"]["prompt"].endswith('"""\n'), repr(item["meta"]["prompt"][-8:])
    assert item["meta"]["prompt"] == _PROMPT
    print("test_humaneval_prompt_keeps_its_trailing_newline: PASS")


def test_humaneval_skips_rows_missing_prompt_or_id():
    rows = "\n".join([
        json.dumps({"task_id": "HumanEval/0", "prompt": _PROMPT, "test": "t",
                    "entry_point": "add_one", "canonical_solution": "x"}),
        json.dumps({"task_id": "HumanEval/1", "prompt": "   \n  "}),   # blank
        json.dumps({"task_id": "", "prompt": _PROMPT}),                 # no id
    ])
    loader = _tmp_loader(HumanEvalLoader, HumanEval__jsonl=rows)
    assert len(loader.parse()) == 1
    print("test_humaneval_skips_rows_missing_prompt_or_id: PASS")


def test_humaneval_task_id_is_filesystem_safe():
    row = json.dumps({"task_id": "HumanEval/61", "prompt": _PROMPT, "test": "t",
                      "entry_point": "add_one", "canonical_solution": "x"})
    loader = _tmp_loader(HumanEvalLoader, HumanEval__jsonl=row)
    item = loader.parse()[0]
    assert item["task_id"] == "HumanEval_61"
    assert item["meta"]["original_task_id"] == "HumanEval/61"
    print("test_humaneval_task_id_is_filesystem_safe: PASS")


# ---------------------------------------------------------------------------
# load(): shuffle before truncating
# ---------------------------------------------------------------------------

class _FakeLoader(DatasetLoader):
    """100 items, label-sorted: first 50 label A, last 50 label B."""
    name = "_fake"
    def download(self): pass
    def parse(self):
        return [TaskItem(task_id=str(i), query="q", gold_answer="g",
                         dataset="_fake", tool_type="none",
                         meta={"label": "A" if i < 50 else "B"})
                for i in range(100)]
    @property
    def tool_type(self): return "none"
    @property
    def description(self): return "fake"


def test_max_samples_slice_is_not_label_sorted():
    # The reason load() shuffles first: Overruling ships all Yes before
    # all No, so an unshuffled head(20) would be 100% one class and the
    # run would silently measure something else entirely.
    items = _FakeLoader(data_dir=Path(tempfile.mkdtemp()), max_samples=20).load()
    labels = {i["meta"]["label"] for i in items}
    assert len(items) == 20
    assert labels == {"A", "B"}, labels
    print("test_max_samples_slice_is_not_label_sorted: PASS")


def test_same_seed_gives_the_same_slice():
    d = Path(tempfile.mkdtemp())
    a = _FakeLoader(data_dir=d, max_samples=20).load(shuffle_seed=7)
    b = _FakeLoader(data_dir=d, max_samples=20).load(shuffle_seed=7)
    assert [i["task_id"] for i in a] == [i["task_id"] for i in b]
    print("test_same_seed_gives_the_same_slice: PASS")


def test_different_seeds_give_different_slices():
    d = Path(tempfile.mkdtemp())
    a = _FakeLoader(data_dir=d, max_samples=20).load(shuffle_seed=1)
    b = _FakeLoader(data_dir=d, max_samples=20).load(shuffle_seed=2)
    assert [i["task_id"] for i in a] != [i["task_id"] for i in b]
    print("test_different_seeds_give_different_slices: PASS")


def test_max_samples_larger_than_dataset_returns_everything():
    items = _FakeLoader(data_dir=Path(tempfile.mkdtemp()), max_samples=10_000).load()
    assert len(items) == 100
    print("test_max_samples_larger_than_dataset_returns_everything: PASS")


# ---------------------------------------------------------------------------
# Integrity + registry
# ---------------------------------------------------------------------------

def test_sha256_check_detects_a_changed_file():
    tmp = Path(tempfile.mkdtemp()) / "f.txt"
    tmp.write_text("hello", encoding="utf-8")
    # sha256("hello")
    good = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert DatasetLoader._check_sha256(tmp, good) is True
    tmp.write_text("hello!", encoding="utf-8")
    assert DatasetLoader._check_sha256(tmp, good) is False
    print("test_sha256_check_detects_a_changed_file: PASS")


def test_unknown_dataset_lists_what_is_available():
    # The error has to name the options; a bare KeyError sends you
    # reading the source to find out what you could have typed.
    try:
        load_dataset("nope")
    except KeyError as exc:
        message = str(exc)
        assert "gsm8k" in message and "humaneval" in message, message
        print("test_unknown_dataset_lists_what_is_available: PASS")
        return
    raise AssertionError("expected KeyError for an unknown dataset")


def test_every_registered_loader_declares_its_metadata():
    entries = list_datasets()
    names = {e["name"] for e in entries}
    assert {"gsm8k", "humaneval", "coqa", "overruling", "headlines"} <= names, names
    for e in entries:
        assert e["tool_type"], e
        assert e["description"], e
    print("test_every_registered_loader_declares_its_metadata: PASS")


def test_tool_necessity_is_not_asserted_by_loaders():
    # It is model-relative and gets labelled by the pilot run. A loader
    # that guessed it would bake an assumption into the ground truth
    # that routing precision/recall is then graded against.
    rows = json.dumps({"question": "What is 6*7?", "answer": "#### 42"})
    item = _tmp_loader(GSM8KLoader, test__jsonl=rows).parse()[0]
    assert "tool_necessity" not in item, item
    print("test_tool_necessity_is_not_asserted_by_loaders: PASS")


# ---------------------------------------------------------------------------
# Headlines cache -- a truncated cache must not be reused for a bigger ask
# ---------------------------------------------------------------------------

def test_no_cache_reports_none():
    assert _cached_rows(Path(tempfile.mkdtemp()) / "missing.jsonl") is None
    print("test_no_cache_reports_none: PASS")


def test_cached_rows_counts_non_blank_lines():
    f = Path(tempfile.mkdtemp()) / "c.jsonl"
    f.write_text("a\n\nb\nc\n", encoding="utf-8")
    assert _cached_rows(f) == 3
    print("test_cached_rows_counts_non_blank_lines: PASS")


def test_a_short_cache_is_detectable():
    # REGRESSION. Unlike the other loaders, headlines caps its DOWNLOAD at
    # max_samples -- 209k rows over a 100-row API is ~2,100 requests. So
    # the cache holds whatever the first call asked for, and reusing it
    # blindly served a later larger request short: ask for 5, then 500, get
    # 5, with nothing saying so. A run 100x smaller than intended still
    # prints a full-looking report.
    f = Path(tempfile.mkdtemp()) / "c.jsonl"
    f.write_text("\n".join(["{}"] * 5) + "\n", encoding="utf-8")
    have = _cached_rows(f)
    assert have == 5
    assert have < 500, "a 5-row cache must not satisfy a 500-row request"
    assert have >= 5, "but it must still satisfy a 5-row request"
    print("test_a_short_cache_is_detectable: PASS")


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
