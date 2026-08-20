"""
test_metrics.py
================

Unit tests for eval_harness.metrics, using small hand-computable examples.
Every expected value here is worked out by hand in the comments -- if a
test fails, the comment tells you what the right answer should be and why.

Run:
    python -m pytest tests/ -v
    (or, without pytest installed: python -m tests.test_metrics)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness.models import (
    TaskRecord, ToolType, ToolNecessity, RoutingDecision, LatencyBreakdown,
)
from eval_harness.metrics import (
    compute_cpst, compute_ece, compute_routing_precision_recall,
    compute_unnecessary_call_rate, compute_missed_call_rate,
    compute_latency_breakdown,
)
from eval_harness.costs import estimate_cost_usd


def make_record(**overrides) -> TaskRecord:
    """Minimal TaskRecord factory with sane defaults, override what you need."""
    defaults = dict(
        task_id="t0",
        dataset="test",
        correct=True,
        tool_necessity=ToolNecessity.NOT_REQUIRED,
        routing_decision=RoutingDecision.DIRECT,
        tool_called=False,
        tool_used=ToolType.NONE,
        confidence_score=0.9,
        model_name="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=50,
        latency=LatencyBreakdown(main_model_ms=500, confidence_ms=20, tool_ms=0),
    )
    defaults.update(overrides)
    return TaskRecord(**defaults)


# ---------------------------------------------------------------------------
# CPST
# ---------------------------------------------------------------------------

def test_cpst_basic():
    # gpt-4o-mini: $0.15/M in, $0.60/M out
    # record cost = (100*0.15 + 50*0.60) / 1e6 = (15 + 30)/1e6 = 4.5e-5
    r1 = make_record(task_id="a", correct=True, prompt_tokens=100, completion_tokens=50)
    r2 = make_record(task_id="b", correct=False, prompt_tokens=100, completion_tokens=50)
    r3 = make_record(task_id="c", correct=True, prompt_tokens=100, completion_tokens=50)

    result = compute_cpst([r1, r2, r3])

    expected_per_record_cost = (100 * 0.15 + 50 * 0.60) / 1_000_000
    expected_total_cost = expected_per_record_cost * 3
    expected_cpst_usd = expected_total_cost / 2  # 2 correct out of 3

    assert result.num_tasks == 3
    assert result.num_correct == 2
    assert abs(result.total_cost_usd - expected_total_cost) < 1e-12
    assert abs(result.cpst_usd - expected_cpst_usd) < 1e-12
    assert result.cpst_tokens == (150 * 3) / 2  # 450 tokens/record * 3 / 2 correct
    print("test_cpst_basic: PASS")


def test_cpst_zero_correct_returns_none():
    r1 = make_record(correct=False)
    result = compute_cpst([r1])
    assert result.cpst_usd is None
    assert result.cpst_tokens is None
    assert result.num_correct == 0
    print("test_cpst_zero_correct_returns_none: PASS")


def test_cost_table_known_value():
    # sanity check the pricing table itself: 1M in + 1M out tokens on
    # gpt-4o-mini should cost exactly 0.15 + 0.60 = 0.75 USD
    cost = estimate_cost_usd("gpt-4o-mini", 1_000_000, 1_000_000)
    assert abs(cost - 0.75) < 1e-9
    print("test_cost_table_known_value: PASS")


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------

def test_ece_perfect_calibration_is_zero():
    # 10 records, confidence == 0.05, 0.15, ..., 0.95 (bin centers),
    # correctness sampled at exactly that rate per bin -> ECE should be ~0
    # Simplify: put all mass in bin covering [0.9,1.0) with confidence 0.95,
    # all correct -> acc=1.0, conf=0.95 -> NOT zero on purpose. Use conf=1.0 instead.
    records = [make_record(task_id=str(i), confidence_score=1.0, correct=True) for i in range(10)]
    result = compute_ece(records, n_bins=10)
    agg = result["aggregate"]
    assert agg.n == 10
    assert abs(agg.ece - 0.0) < 1e-9  # conf=1.0, acc=1.0 in that bin -> |1-1|=0
    print("test_ece_perfect_calibration_is_zero: PASS")


def test_ece_maximally_miscalibrated():
    # confidence = 1.0 (says "certain") but always wrong -> |acc - conf| = |0 - 1| = 1
    records = [make_record(task_id=str(i), confidence_score=1.0, correct=False) for i in range(10)]
    result = compute_ece(records, n_bins=10)
    agg = result["aggregate"]
    assert abs(agg.ece - 1.0) < 1e-9
    print("test_ece_maximally_miscalibrated: PASS")


def test_ece_skips_records_without_confidence_or_correct():
    r1 = make_record(task_id="a", confidence_score=None, correct=True)
    r2 = make_record(task_id="b", confidence_score=0.8, correct=None)
    r3 = make_record(task_id="c", confidence_score=0.8, correct=True)
    result = compute_ece([r1, r2, r3], n_bins=10)
    assert result["aggregate"].n == 1  # only r3 is eligible
    print("test_ece_skips_records_without_confidence_or_correct: PASS")


def test_ece_grouping():
    retrieval_records = [
        make_record(task_id=f"r{i}", confidence_score=1.0, correct=True, tool_used=ToolType.RETRIEVAL)
        for i in range(5)
    ]
    calc_records = [
        make_record(task_id=f"c{i}", confidence_score=1.0, correct=False, tool_used=ToolType.CALCULATOR)
        for i in range(5)
    ]
    result = compute_ece(
        retrieval_records + calc_records, n_bins=10, group_by=lambda r: r.tool_used.value
    )
    assert abs(result["retrieval"].ece - 0.0) < 1e-9
    assert abs(result["calculator"].ece - 1.0) < 1e-9
    # aggregate pools everything: conf=1.0 always, correct 5/10 -> |0.5 - 1.0| = 0.5
    assert abs(result["aggregate"].ece - 0.5) < 1e-9
    print("test_ece_grouping: PASS")


# ---------------------------------------------------------------------------
# Routing precision / recall
# ---------------------------------------------------------------------------

def test_routing_precision_recall_confusion_matrix():
    records = [
        # TP: needed tool, called it
        make_record(task_id="tp1", tool_necessity=ToolNecessity.REQUIRED, tool_called=True),
        make_record(task_id="tp2", tool_necessity=ToolNecessity.REQUIRED, tool_called=True),
        # FP: didn't need tool, called it anyway
        make_record(task_id="fp1", tool_necessity=ToolNecessity.NOT_REQUIRED, tool_called=True),
        # FN: needed tool, didn't call it
        make_record(task_id="fn1", tool_necessity=ToolNecessity.REQUIRED, tool_called=False),
        # TN: didn't need tool, didn't call it
        make_record(task_id="tn1", tool_necessity=ToolNecessity.NOT_REQUIRED, tool_called=False),
        make_record(task_id="tn2", tool_necessity=ToolNecessity.NOT_REQUIRED, tool_called=False),
    ]
    result = compute_routing_precision_recall(records)
    assert (result.tp, result.fp, result.fn, result.tn) == (2, 1, 1, 2)
    # precision = tp/(tp+fp) = 2/3
    assert abs(result.precision - 2 / 3) < 1e-9
    # recall = tp/(tp+fn) = 2/3
    assert abs(result.recall - 2 / 3) < 1e-9
    assert abs(result.f1 - 2 / 3) < 1e-9
    print("test_routing_precision_recall_confusion_matrix: PASS")


def test_routing_excludes_ambiguous_by_default():
    records = [
        make_record(task_id="amb1", tool_necessity=ToolNecessity.AMBIGUOUS, tool_called=True),
        make_record(task_id="tp1", tool_necessity=ToolNecessity.REQUIRED, tool_called=True),
    ]
    result = compute_routing_precision_recall(records)
    assert result.n_excluded_ambiguous == 1
    assert result.tp == 1
    assert (result.fp, result.fn, result.tn) == (0, 0, 0)
    print("test_routing_excludes_ambiguous_by_default: PASS")


def test_unnecessary_and_missed_call_rates():
    records = [
        make_record(task_id="fp1", tool_necessity=ToolNecessity.NOT_REQUIRED, tool_called=True),
        make_record(task_id="tn1", tool_necessity=ToolNecessity.NOT_REQUIRED, tool_called=False),
        make_record(task_id="tn2", tool_necessity=ToolNecessity.NOT_REQUIRED, tool_called=False),
        make_record(task_id="tn3", tool_necessity=ToolNecessity.NOT_REQUIRED, tool_called=False),
        make_record(task_id="fn1", tool_necessity=ToolNecessity.REQUIRED, tool_called=False),
        make_record(task_id="tp1", tool_necessity=ToolNecessity.REQUIRED, tool_called=True),
    ]
    # unnecessary rate = FP / (FP+TN) = 1/4
    assert abs(compute_unnecessary_call_rate(records) - 0.25) < 1e-9
    # missed rate = FN / (FN+TP) = 1/2
    assert abs(compute_missed_call_rate(records) - 0.5) < 1e-9
    print("test_unnecessary_and_missed_call_rates: PASS")


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

def test_latency_breakdown_direct_vs_tool():
    r1 = make_record(
        task_id="d1", routing_decision=RoutingDecision.DIRECT,
        latency=LatencyBreakdown(main_model_ms=100, confidence_ms=10, tool_ms=0),
    )
    r2 = make_record(
        task_id="d2", routing_decision=RoutingDecision.DIRECT,
        latency=LatencyBreakdown(main_model_ms=300, confidence_ms=10, tool_ms=0),
    )
    r3 = make_record(
        task_id="t1", routing_decision=RoutingDecision.TOOL,
        latency=LatencyBreakdown(main_model_ms=100, confidence_ms=10, tool_ms=1000),
    )
    result = compute_latency_breakdown([r1, r2, r3])
    assert result.direct_only.n == 2
    assert abs(result.direct_only.mean_ms - 210) < 1e-6  # (110+310)/2
    assert result.tool_only.n == 1
    assert abs(result.tool_only.mean_ms - 1110) < 1e-6  # 100+10+1000
    assert result.overall.n == 3
    print("test_latency_breakdown_direct_vs_tool: PASS")


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
