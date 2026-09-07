"""
router.py
=========

The routing decision itself: confidence in, DIRECT or TOOL out.

Phase 3. Everything before this produced a number -- confidence.py
estimates P(the direct answer is correct), metrics.py grades decisions
after the fact. This is the piece in between, and it is deliberately the
smallest module in the harness: one comparison, plus the two questions
that comparison raises.

Question one: what does a threshold mean at the boundary? TOOL iff
``confidence < threshold``, strictly. A task sitting exactly at 0.7 with
threshold 0.7 answers directly. This matters more than it looks --
self-consistency confidences are quantized to k-ths (k=5 samples can
only ever return 0.0, 0.2, 0.4, 0.6, 0.8, 1.0), so exact ties with a
round threshold are common rather than measure-zero, and a sweep that
flips its convention halfway produces a discontinuous curve.

Question two: what happens when there is no confidence? An estimator
returns None when the signal was not available -- logprobs were not
requested, every sample failed to parse, the verifier call errored. The
hybrid combiner already treats a missing signal as an absent opinion
rather than a vote of no confidence; the router cannot be so relaxed,
because it has to emit a decision either way. Default is to escalate:
no evidence of confidence is not evidence of confidence, and the
expensive failure mode of this whole project is a confident-and-wrong
direct answer, not a wasted tool call. Callers running a
cost-at-all-costs ablation can flip it with ``on_missing="direct"``.

NaN counts as missing, and that is a load-bearing line rather than
defensive noise: ``float("nan") < 0.7`` is False, so a NaN confidence
that fell through would silently route DIRECT -- the exact direction the
default is chosen to avoid.

--- Threshold selection ------------------------------------------------

``sweep_thresholds`` re-decides routing for a set of already-logged
records at each candidate threshold and grades each one, which is how
the operating point gets picked from a pilot run instead of by taste.
It reuses metrics.py rather than recounting a confusion matrix here:
the sweep builds a copy of each record with the counterfactual decision
applied and hands the list to the existing metric functions, so a
threshold-0.7 row of a sweep and a real run at 0.7 are computed by the
same code.

What the sweep deliberately does NOT report is CPST, and the omission is
the honest part. Cost per successful task depends on whether the task
came out correct, and correctness under a decision the system did not
actually take is not in the log -- a task answered directly and graded
wrong has no recorded answer for the tool path it never took. Every
number here is a function of ``tool_necessity`` and the decision alone,
both of which survive the counterfactual. Getting a real CPST curve
takes running the pipeline at each threshold, not re-reading one run.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Union

from .models import TaskRecord, RoutingDecision, ToolType
from .metrics import (
    compute_routing_precision_recall,
    compute_unnecessary_call_rate,
    compute_missed_call_rate,
)

# What to do with a task whose confidence estimator returned nothing.
MISSING_POLICIES = ("tool", "direct")

# 0.00, 0.05, ... 1.00 -- fine enough to see the knee of the curve,
# coarse enough to print.
DEFAULT_THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(21))


def route(
    confidence: Optional[float],
    threshold: float = 0.7,
    *,
    on_missing: str = "tool",
) -> RoutingDecision:
    """
    TOOL iff ``confidence < threshold``. See the module docstring for why
    the comparison is strict and why missing escalates by default.
    """
    if on_missing not in MISSING_POLICIES:
        raise ValueError(f"on_missing must be one of {MISSING_POLICIES}, got {on_missing!r}")

    # NaN != NaN; a NaN confidence is missing, not low.
    if confidence is None or confidence != confidence:
        return RoutingDecision.TOOL if on_missing == "tool" else RoutingDecision.DIRECT

    return RoutingDecision.TOOL if confidence < threshold else RoutingDecision.DIRECT


def route_record(
    record: TaskRecord,
    threshold: Optional[float] = None,
    *,
    tool: ToolType = ToolType.NONE,
    on_missing: str = "tool",
) -> TaskRecord:
    """
    A copy of ``record`` with the routing fields set from its own
    confidence. ``threshold=None`` uses the record's own
    ``confidence_threshold``, which is what the sweep overrides.

    ``tool`` is which tool a TOOL decision would call. The router does
    not pick tools -- that is the task's category, and the caller knows
    it -- so it defaults to NONE and the caller passes the real one.
    """
    if threshold is None:
        threshold = record.confidence_threshold

    decision = route(record.confidence_score, threshold, on_missing=on_missing)
    called = decision == RoutingDecision.TOOL

    return replace(
        record,
        routing_decision=decision,
        tool_called=called,
        tool_used=tool if called else ToolType.NONE,
        confidence_threshold=threshold,
    )


@dataclass
class ThresholdPoint:
    """One row of a sweep: what routing would have looked like at this cut."""
    threshold: float
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    unnecessary_call_rate: Optional[float]
    missed_call_rate: Optional[float]
    n_tool: int
    n_direct: int
    n_missing_confidence: int


def sweep_thresholds(
    records: Sequence[TaskRecord],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    *,
    on_missing: str = "tool",
) -> List[ThresholdPoint]:
    """
    Grade every candidate threshold against already-logged records.

    No CPST column -- see the module docstring. The metrics here depend
    only on the ground-truth necessity label and the decision, so they
    survive being recomputed counterfactually.
    """
    n_missing = sum(
        1 for r in records
        if r.confidence_score is None or r.confidence_score != r.confidence_score
    )

    points = []
    for threshold in thresholds:
        # ponytail: copies every record per threshold, O(n*|thresholds|).
        # Reuses metrics.py instead of a second confusion matrix, which is
        # worth more than the copies at pilot-run sizes. Count inline if a
        # sweep ever runs over a log big enough to notice.
        rerouted = [route_record(r, threshold, on_missing=on_missing) for r in records]
        pr = compute_routing_precision_recall(rerouted)

        points.append(ThresholdPoint(
            threshold=threshold,
            precision=pr.precision,
            recall=pr.recall,
            f1=pr.f1,
            unnecessary_call_rate=compute_unnecessary_call_rate(rerouted),
            missed_call_rate=compute_missed_call_rate(rerouted),
            n_tool=sum(1 for r in rerouted if r.tool_called),
            n_direct=sum(1 for r in rerouted if not r.tool_called),
            n_missing_confidence=n_missing,
        ))

    return points


def select_threshold(
    records: Sequence[TaskRecord],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    *,
    objective: Union[str, Callable[[ThresholdPoint], Optional[float]]] = "f1",
    on_missing: str = "tool",
) -> Optional[ThresholdPoint]:
    """
    The best-scoring point of a sweep. ``objective`` is "f1" or any
    callable taking a ThresholdPoint -- a callable covers every
    cost-weighted trade-off worth trying without this function growing a
    config object for each one, e.g.::

        select_threshold(records, objective=lambda p: -p.unnecessary_call_rate
                         if p.missed_call_rate <= 0.1 else None)

    Points the objective scores as None are skipped (undefined metric, or
    a constraint the caller rejected). Returns None if none survive.
    Ties go to the HIGHER threshold, which is the cheaper end: same
    routing quality, fewer tool calls.
    """
    score = (lambda p: p.f1) if objective == "f1" else objective
    if not callable(score):
        raise ValueError(f"objective must be 'f1' or a callable, got {objective!r}")

    best: Optional[ThresholdPoint] = None
    best_score: Optional[float] = None
    for point in sweep_thresholds(records, thresholds, on_missing=on_missing):
        value = score(point)
        if value is None:
            continue
        if best_score is None or value > best_score or (value == best_score and point.threshold > best.threshold):
            best, best_score = point, value

    return best
