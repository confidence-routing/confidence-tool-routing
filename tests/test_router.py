"""
test_router.py
==============

Unit tests for eval_harness.router, on small hand-computable examples.
Every expected value is worked out in the comments -- if a test fails,
the comment says what the right answer is and why.

Run:
    python -m pytest tests/ -v
    (or, without pytest installed: python -m tests.test_router)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness.models import (
    TaskRecord, ToolType, ToolNecessity, RoutingDecision, LatencyBreakdown,
)
from eval_harness.router import (
    route, route_record, sweep_thresholds, select_threshold,
    DEFAULT_THRESHOLDS,
)


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
        confidence_threshold=0.7,
        latency=LatencyBreakdown(main_model_ms=500, confidence_ms=20, tool_ms=0),
    )
    defaults.update(overrides)
    return TaskRecord(**defaults)


def test_route_below_threshold_calls_tool():
    assert route(0.3, 0.7) == RoutingDecision.TOOL
    assert route(0.0, 0.7) == RoutingDecision.TOOL
    print("test_route_below_threshold_calls_tool: PASS")


def test_route_above_threshold_answers_direct():
    assert route(0.9, 0.7) == RoutingDecision.DIRECT
    assert route(1.0, 0.7) == RoutingDecision.DIRECT
    print("test_route_above_threshold_answers_direct: PASS")


def test_route_exactly_at_threshold_answers_direct():
    # Strict <: 0.7 is not below 0.7. Load-bearing for quantized
    # self-consistency scores (k=5 -> 0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    # where exact ties with a round threshold are common.
    assert route(0.7, 0.7) == RoutingDecision.DIRECT
    assert route(0.2, 0.2) == RoutingDecision.DIRECT
    print("test_route_exactly_at_threshold_answers_direct: PASS")


def test_route_missing_confidence_escalates_by_default():
    assert route(None, 0.7) == RoutingDecision.TOOL
    assert route(None, 0.7, on_missing="direct") == RoutingDecision.DIRECT
    print("test_route_missing_confidence_escalates_by_default: PASS")


def test_route_nan_is_missing_not_low():
    # nan < 0.7 is False, so an unguarded comparison routes DIRECT --
    # the one direction the missing-signal default exists to prevent.
    nan = float("nan")
    assert route(nan, 0.7) == RoutingDecision.TOOL
    assert route(nan, 0.7, on_missing="direct") == RoutingDecision.DIRECT
    print("test_route_nan_is_missing_not_low: PASS")


def test_route_rejects_unknown_missing_policy():
    try:
        route(0.5, 0.7, on_missing="escalate")
    except ValueError:
        print("test_route_rejects_unknown_missing_policy: PASS")
        return
    raise AssertionError("expected ValueError for unknown on_missing policy")


def test_route_record_applies_decision_and_tool():
    r = make_record(confidence_score=0.4, confidence_threshold=0.7)
    out = route_record(r, tool=ToolType.CALCULATOR)

    assert out.routing_decision == RoutingDecision.TOOL
    assert out.tool_called is True
    assert out.tool_used == ToolType.CALCULATOR
    # original is untouched -- route_record returns a copy
    assert r.tool_called is False
    assert r.tool_used == ToolType.NONE
    print("test_route_record_applies_decision_and_tool: PASS")


def test_route_record_direct_clears_tool():
    # A DIRECT decision must leave tool_used at NONE even though the
    # caller offered one, or the cost model bills a tool never called.
    r = make_record(confidence_score=0.95, confidence_threshold=0.7)
    out = route_record(r, tool=ToolType.WEB_SEARCH)

    assert out.routing_decision == RoutingDecision.DIRECT
    assert out.tool_called is False
    assert out.tool_used == ToolType.NONE
    print("test_route_record_direct_clears_tool: PASS")


def test_route_record_defaults_to_records_own_threshold():
    r = make_record(confidence_score=0.6, confidence_threshold=0.8)
    assert route_record(r).tool_called is True          # 0.6 < 0.8
    assert route_record(r, 0.5).tool_called is False    # 0.6 >= 0.5
    assert route_record(r, 0.5).confidence_threshold == 0.5  # override recorded
    print("test_route_record_defaults_to_records_own_threshold: PASS")


def _pr_fixture():
    """Four records, one per confusion-matrix cell at threshold 0.7.

        conf 0.2, REQUIRED     -> tool called, needed      TP
        conf 0.3, NOT_REQUIRED -> tool called, not needed  FP
        conf 0.9, REQUIRED     -> no tool, needed          FN
        conf 0.8, NOT_REQUIRED -> no tool, not needed      TN
    """
    return [
        make_record(task_id="tp", confidence_score=0.2, tool_necessity=ToolNecessity.REQUIRED),
        make_record(task_id="fp", confidence_score=0.3, tool_necessity=ToolNecessity.NOT_REQUIRED),
        make_record(task_id="fn", confidence_score=0.9, tool_necessity=ToolNecessity.REQUIRED),
        make_record(task_id="tn", confidence_score=0.8, tool_necessity=ToolNecessity.NOT_REQUIRED),
    ]


def test_sweep_grades_each_threshold():
    points = {p.threshold: p for p in sweep_thresholds(_pr_fixture(), [0.0, 0.7, 1.0])}

    # 0.0: nothing is below 0.0, so every task answers directly.
    assert points[0.0].n_tool == 0 and points[0.0].n_direct == 4
    assert points[0.0].precision is None          # no calls -> undefined
    assert points[0.0].missed_call_rate == 1.0    # both REQUIRED tasks missed
    assert points[0.0].unnecessary_call_rate == 0.0

    # 0.7: the fixture's designed confusion matrix, TP=FP=FN=TN=1.
    # precision = 1/(1+1) = 0.5, recall = 1/(1+1) = 0.5, f1 = 0.5
    assert points[0.7].n_tool == 2 and points[0.7].n_direct == 2
    assert points[0.7].precision == 0.5
    assert points[0.7].recall == 0.5
    assert points[0.7].f1 == 0.5
    assert points[0.7].unnecessary_call_rate == 0.5   # 1 of 2 NOT_REQUIRED
    assert points[0.7].missed_call_rate == 0.5        # 1 of 2 REQUIRED

    # 1.0: every confidence is below 1.0, so everything escalates.
    # precision = 2/4 = 0.5, recall = 2/2 = 1.0
    assert points[1.0].n_tool == 4 and points[1.0].n_direct == 0
    assert points[1.0].recall == 1.0
    assert points[1.0].unnecessary_call_rate == 1.0
    assert points[1.0].missed_call_rate == 0.0
    print("test_sweep_grades_each_threshold: PASS")


def test_sweep_counts_missing_confidence():
    records = _pr_fixture() + [make_record(task_id="none", confidence_score=None)]
    for point in sweep_thresholds(records, [0.0, 0.7]):
        assert point.n_missing_confidence == 1
    # on_missing="tool" means the missing-confidence task escalates even
    # at threshold 0.0, where nothing else does.
    assert sweep_thresholds(records, [0.0])[0].n_tool == 1
    assert sweep_thresholds(records, [0.0], on_missing="direct")[0].n_tool == 0
    print("test_sweep_counts_missing_confidence: PASS")


def test_sweep_default_thresholds_cover_unit_interval():
    points = sweep_thresholds(_pr_fixture())
    assert len(points) == len(DEFAULT_THRESHOLDS) == 21
    assert points[0].threshold == 0.0 and points[-1].threshold == 1.0
    # monotone: raising the threshold can only add tool calls
    counts = [p.n_tool for p in points]
    assert counts == sorted(counts)
    print("test_sweep_default_thresholds_cover_unit_interval: PASS")


def test_select_threshold_maximizes_f1():
    # Confidence is perfectly separable here: REQUIRED tasks score 0.1/0.2,
    # NOT_REQUIRED score 0.8/0.9. Any cut in (0.2, 0.8] gives f1 = 1.0.
    records = [
        make_record(task_id="a", confidence_score=0.1, tool_necessity=ToolNecessity.REQUIRED),
        make_record(task_id="b", confidence_score=0.2, tool_necessity=ToolNecessity.REQUIRED),
        make_record(task_id="c", confidence_score=0.8, tool_necessity=ToolNecessity.NOT_REQUIRED),
        make_record(task_id="d", confidence_score=0.9, tool_necessity=ToolNecessity.NOT_REQUIRED),
    ]
    best = select_threshold(records)
    assert best.f1 == 1.0
    # 0.25 through 0.80 all score 1.0; ties go to the highest, and 0.80
    # is the last cut that still leaves c and d direct (strict <).
    assert best.threshold == 0.8
    print("test_select_threshold_maximizes_f1: PASS")


def test_select_threshold_accepts_callable_objective():
    records = _pr_fixture()
    # "cheapest routing that misses no more than half the required tasks"
    best = select_threshold(
        records,
        objective=lambda p: (
            -p.unnecessary_call_rate
            if p.missed_call_rate is not None and p.missed_call_rate <= 0.5
            else None
        ),
    )
    # missed <= 0.5 needs the 0.2-confidence REQUIRED task to escalate,
    # so threshold > 0.2; the cheapest such cut calls no unnecessary
    # tools, which holds up to 0.3 (the FP task's confidence).
    assert best.missed_call_rate == 0.5
    assert best.unnecessary_call_rate == 0.0
    assert best.threshold == 0.3
    print("test_select_threshold_accepts_callable_objective: PASS")


def test_select_threshold_returns_none_when_objective_rejects_all():
    assert select_threshold(_pr_fixture(), objective=lambda p: None) is None
    print("test_select_threshold_returns_none_when_objective_rejects_all: PASS")


def test_select_threshold_rejects_bad_objective():
    try:
        select_threshold(_pr_fixture(), objective="accuracy")
    except ValueError:
        print("test_select_threshold_rejects_bad_objective: PASS")
        return
    raise AssertionError("expected ValueError for unknown objective")


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
