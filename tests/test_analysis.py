"""
test_analysis.py
================

Unit tests for eval_harness.analysis -- the cross-run comparisons.

The transfer-matrix tests matter most. That matrix is the project's
headline result, so the fixtures below are built so the right answer is
known by construction: one pair of datasets whose optimal thresholds
coincide (transfer should cost nothing) and one pair whose optimal
thresholds are far apart (transfer should be visibly bad). A matrix that
cannot tell those apart is useless, so both directions are pinned.

Run:
    python -m pytest tests/ -v
    (or, without pytest installed: python -m tests.test_analysis)
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness.analysis import (
    breakeven_tool_fee, compare_runs, load_runs, transfer_matrix,
)
from eval_harness.confidence import ConfidenceMethod
from eval_harness.logger import RunLogger
from eval_harness.models import (
    RoutingDecision, TaskRecord, ToolNecessity, ToolType,
)


def rec(task_id, dataset, conf, necessity, *, correct=True,
        method=ConfidenceMethod.ENTROPY, tool_called=False,
        conf_prompt=0, conf_completion=0, tool_prompt=0, tool_completion=0,
        tool_fee=0.0, model="gpt-4o-mini"):
    return TaskRecord(
        task_id=task_id, dataset=dataset, confidence_score=conf,
        confidence_method=method, tool_necessity=necessity, correct=correct,
        tool_called=tool_called,
        tool_used=ToolType.CALCULATOR if tool_called else ToolType.NONE,
        routing_decision=RoutingDecision.TOOL if tool_called else RoutingDecision.DIRECT,
        model_name=model, prompt_tokens=100, completion_tokens=10,
        confidence_prompt_tokens=conf_prompt,
        confidence_completion_tokens=conf_completion,
        confidence_model_name=model if conf_prompt else "",
        tool_prompt_tokens=tool_prompt, tool_completion_tokens=tool_completion,
        tool_api_cost_usd=tool_fee,
    )


REQ, NOT = ToolNecessity.REQUIRED, ToolNecessity.NOT_REQUIRED


def _separable(dataset, low, high):
    """Required tasks score `low`, not-required score `high`. Any cut
    strictly between them is perfect."""
    return ([rec(f"{dataset}_r{i}", dataset, low, REQ) for i in range(5)] +
            [rec(f"{dataset}_n{i}", dataset, high, NOT) for i in range(5)])


# ---------------------------------------------------------------------------
# Loading and grouping
# ---------------------------------------------------------------------------

def test_runs_group_by_record_fields_not_filename():
    # A log named for one thing may hold another; dataset and
    # confidence_method are fields the pipeline set, a filename is not.
    d = Path(tempfile.mkdtemp())
    with RunLogger("misleading_name", log_dir=str(d), resume=True) as lg:
        lg.log(rec("a", "gsm8k", 0.9, NOT, method=ConfidenceMethod.ENTROPY))
        lg.log(rec("b", "coqa", 0.2, REQ, method=ConfidenceMethod.HYBRID))
    grouped = load_runs([str(d / "misleading_name.jsonl")])
    assert set(grouped) == {("gsm8k", "entropy"), ("coqa", "hybrid")}
    print("test_runs_group_by_record_fields_not_filename: PASS")


# ---------------------------------------------------------------------------
# The estimator table
# ---------------------------------------------------------------------------

def test_confidence_cost_is_broken_out():
    # The column the project turns on: without it, free entropy and
    # k-sample self-consistency are indistinguishable in a table.
    free = [rec("a", "gsm8k", 0.9, NOT)]
    paid = [rec("b", "gsm8k", 0.9, NOT, method=ConfidenceMethod.SELF_CONSISTENCY,
                conf_prompt=400, conf_completion=40)]
    rows = {r.method: r for r in compare_runs(
        {("gsm8k", "entropy"): free, ("gsm8k", "self_consistency"): paid})}
    assert rows["entropy"].confidence_cost_usd == 0.0
    assert rows["self_consistency"].confidence_cost_usd > 0.0
    assert rows["self_consistency"].confidence_cost_share > 0.0
    print("test_confidence_cost_is_broken_out: PASS")


def test_tool_call_rate_is_reported():
    records = [rec("a", "gsm8k", 0.2, REQ, tool_called=True),
               rec("b", "gsm8k", 0.9, NOT),
               rec("c", "gsm8k", 0.9, NOT),
               rec("d", "gsm8k", 0.1, REQ, tool_called=True)]
    row = compare_runs({("gsm8k", "entropy"): records})[0]
    assert row.tool_call_rate == 0.5
    assert row.n == 4
    print("test_tool_call_rate_is_reported: PASS")


# ---------------------------------------------------------------------------
# Transfer matrix -- the headline
# ---------------------------------------------------------------------------

def test_matching_datasets_transfer_without_loss():
    # Both separable at the same place, so one dataset's threshold is also
    # optimal for the other and degradation must be zero.
    a = _separable("alpha", 0.1, 0.9)
    b = _separable("beta", 0.1, 0.9)
    cells = {(c.tuned_on, c.evaluated_on): c
             for c in transfer_matrix({"alpha": a, "beta": b})}
    cross = cells[("alpha", "beta")]
    assert cross.f1 == 1.0, cross
    assert cross.degradation == 0.0, cross
    print("test_matching_datasets_transfer_without_loss: PASS")


def test_mismatched_datasets_show_degradation():
    # alpha separates at ~0.1/0.4, beta at ~0.6/0.95. alpha's threshold is
    # far below beta's required scores, so applying it leaves beta's
    # required tasks answered directly -- recall collapses.
    #
    # This is the finding the matrix exists to detect: a threshold that is
    # a property of the task rather than the model.
    a = _separable("alpha", 0.1, 0.4)
    b = _separable("beta", 0.6, 0.95)
    cells = {(c.tuned_on, c.evaluated_on): c
             for c in transfer_matrix({"alpha": a, "beta": b})}

    own = cells[("beta", "beta")]
    borrowed = cells[("alpha", "beta")]
    assert own.f1 == 1.0, own
    assert borrowed.f1 in (None, 0.0) or borrowed.f1 < own.f1, borrowed
    assert borrowed.degradation is None or borrowed.degradation > 0.5, borrowed
    print("test_mismatched_datasets_show_degradation: PASS")


def test_diagonal_is_the_in_domain_best():
    a = _separable("alpha", 0.1, 0.9)
    cells = [c for c in transfer_matrix({"alpha": a})]
    assert len(cells) == 1
    assert cells[0].tuned_on == cells[0].evaluated_on == "alpha"
    assert cells[0].degradation == 0.0
    print("test_diagonal_is_the_in_domain_best: PASS")


def test_matrix_reports_no_cpst_column():
    # Deliberate. Routing survives counterfactual recomputation because
    # tool_necessity and the decision are both logged; correctness under a
    # path never taken does not, so a CPST column here would be invented.
    cell = transfer_matrix({"alpha": _separable("alpha", 0.1, 0.9)})[0]
    assert not hasattr(cell, "cpst_usd")
    assert not any("cpst" in f.lower() for f in vars(cell))
    print("test_matrix_reports_no_cpst_column: PASS")


# ---------------------------------------------------------------------------
# Break-even on tool price
# ---------------------------------------------------------------------------

def test_free_estimator_always_pays_off():
    # Token entropy spends nothing, so every task answered directly is a
    # net saving and no fee is needed to justify routing.
    records = [rec("a", "gsm8k", 0.9, NOT),
               rec("b", "gsm8k", 0.1, REQ, tool_called=True,
                   tool_prompt=100, tool_completion=20)]
    out = breakeven_tool_fee(records)
    assert out.confidence_tokens == 0
    assert out.pays_off_now is True
    assert out.breakeven_fee_tokens == 0.0
    print("test_free_estimator_always_pays_off: PASS")


def test_expensive_estimator_needs_a_dear_tool():
    # A heavy estimator on every task against a cheap tool. 8 direct tasks
    # each avoided ~120 tool tokens = 960 saved; the estimator spent
    # 4400 x 10 = 44000. Routing loses, and the fee the tool would need to
    # carry comes back positive. This is the arithmetic the project rests on.
    records = ([rec(f"d{i}", "gsm8k", 0.9, NOT, conf_prompt=4000,
                    conf_completion=400) for i in range(8)] +
               [rec(f"t{i}", "gsm8k", 0.1, REQ, tool_called=True,
                    tool_prompt=100, tool_completion=20,
                    conf_prompt=4000, conf_completion=400) for i in range(2)])
    out = breakeven_tool_fee(records)
    assert out.confidence_tokens == 44000, out.confidence_tokens
    assert out.pays_off_now is False
    assert out.breakeven_fee_tokens > 0, out
    print("test_expensive_estimator_needs_a_dear_tool: PASS")


def test_no_direct_answers_means_nothing_was_saved():
    # Every task escalated, so routing bought nothing and no fee makes it
    # cheaper -- None rather than a number implying a trade-off exists.
    records = [rec("a", "gsm8k", 0.1, REQ, tool_called=True,
                   tool_prompt=100, tool_completion=20)]
    out = breakeven_tool_fee(records)
    assert out.direct_calls == 0
    assert out.breakeven_fee_tokens is None
    print("test_no_direct_answers_means_nothing_was_saved: PASS")


def test_an_unpriced_model_still_analyses():
    # REGRESSION. The first run of the real matrix died here: a locally
    # served model has no rate, estimate_cost_usd rightly refuses to
    # invent one, and the analysis raised KeyError instead of reporting
    # what it could. Tokens need no rate, and the token figures are what
    # the estimator comparison actually rests on -- so dollars go None
    # and nothing else is lost.
    records = [rec("a", "gsm8k", 0.9, NOT, model="llama3.2:3b",
                   conf_prompt=400, conf_completion=40),
               rec("b", "gsm8k", 0.1, REQ, model="llama3.2:3b",
                   tool_called=True, tool_prompt=100, tool_completion=20)]
    rows = compare_runs({("gsm8k", "entropy"): records})
    row = rows[0]
    assert row.confidence_cost_usd is None, "no rate means no dollar figure"
    assert row.confidence_tokens == 440, row.confidence_tokens
    assert row.confidence_token_share > 0
    assert row.cpst_usd is None
    assert row.cpst_tokens is not None, "token CPST needs no rate"

    out = breakeven_tool_fee(records)
    assert out.routed_spend_usd is None
    assert out.confidence_tokens == 440
    assert out.breakeven_fee_tokens is not None, "token break-even needs no rate"
    print("test_an_unpriced_model_still_analyses: PASS")


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
