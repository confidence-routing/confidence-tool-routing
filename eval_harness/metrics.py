"""
metrics.py
==========

All metrics computed from a list of TaskRecord. Every function here is a
pure function: records in, numbers out. No I/O, no side effects, so these
are trivial to unit test and to reuse in notebooks.

Metrics implemented:
    - compute_cpst                    Cost Per Successful Task
    - compute_ece                     Expected Calibration Error (+ optional grouping)
    - compute_routing_precision_recall  Tool-call decision precision/recall/F1
    - compute_unnecessary_call_rate   False-positive tool calls
    - compute_missed_call_rate        False-negative (missed) tool calls
    - compute_latency_breakdown       Mean/median/p95 latency, direct vs tool
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .models import TaskRecord, ToolNecessity, RoutingDecision
from .costs import estimate_record_cost_usd


# ---------------------------------------------------------------------------
# CPST — Cost Per Successful Task
# ---------------------------------------------------------------------------

@dataclass
class CPSTResult:
    cpst_usd: Optional[float]       # total USD cost / num_correct
    cpst_tokens: Optional[float]    # total tokens / num_correct
    total_cost_usd: float
    total_tokens: int
    num_tasks: int
    num_correct: int
    success_rate: float


def compute_cpst(records: Sequence[TaskRecord]) -> CPSTResult:
    """
    CPST = (tokens + API cost) / correct tasks.

    We report the USD variant (total cost in dollars, including both
    token-based LLM cost and any flat tool-API cost, divided by number of
    correct tasks) as the primary number, since that's the actionable
    business metric. The pure token-count variant is reported alongside
    for transparency and for comparing against token-only metrics in
    other papers (e.g. DataGovBench's T*).

    Tasks with num_correct == 0 return cpst_usd = None / cpst_tokens = None
    (undefined — avoid division by zero, don't silently report inf).
    """
    total_cost_usd = 0.0
    total_tokens = 0
    num_correct = 0

    for r in records:
        total_cost_usd += estimate_record_cost_usd(r)
        total_tokens += r.total_prompt_tokens + r.total_completion_tokens
        if r.correct:
            num_correct += 1

    num_tasks = len(records)
    success_rate = num_correct / num_tasks if num_tasks else 0.0

    cpst_usd = total_cost_usd / num_correct if num_correct else None
    cpst_tokens = total_tokens / num_correct if num_correct else None

    return CPSTResult(
        cpst_usd=cpst_usd,
        cpst_tokens=cpst_tokens,
        total_cost_usd=total_cost_usd,
        total_tokens=total_tokens,
        num_tasks=num_tasks,
        num_correct=num_correct,
        success_rate=success_rate,
    )


# ---------------------------------------------------------------------------
# ECE — Expected Calibration Error
# ---------------------------------------------------------------------------

@dataclass
class ECEResult:
    ece: Optional[float]
    n: int
    bin_accuracy: List[float]
    bin_confidence: List[float]
    bin_counts: List[int]


def _ece_single_group(
    confidences: List[float], corrects: List[bool], n_bins: int
) -> ECEResult:
    n = len(confidences)
    if n == 0:
        return ECEResult(ece=None, n=0, bin_accuracy=[], bin_confidence=[], bin_counts=[])

    edges = [i / n_bins for i in range(n_bins + 1)]
    bin_correct = [0] * n_bins
    bin_conf_sum = [0.0] * n_bins
    bin_count = [0] * n_bins

    for conf, correct in zip(confidences, corrects):
        conf = min(max(conf, 0.0), 1.0)
        # last bin is closed on the right: [.., 1.0]
        idx = min(int(conf * n_bins), n_bins - 1)
        bin_count[idx] += 1
        bin_conf_sum[idx] += conf
        bin_correct[idx] += int(bool(correct))

    ece = 0.0
    bin_accuracy: List[float] = []
    bin_confidence: List[float] = []
    for i in range(n_bins):
        if bin_count[i] == 0:
            bin_accuracy.append(0.0)
            bin_confidence.append(0.0)
            continue
        acc = bin_correct[i] / bin_count[i]
        conf = bin_conf_sum[i] / bin_count[i]
        bin_accuracy.append(acc)
        bin_confidence.append(conf)
        ece += (bin_count[i] / n) * abs(acc - conf)

    return ECEResult(
        ece=ece, n=n, bin_accuracy=bin_accuracy, bin_confidence=bin_confidence, bin_counts=bin_count
    )


def compute_ece(
    records: Sequence[TaskRecord],
    n_bins: int = 10,
    group_by: Optional[Callable[[TaskRecord], str]] = None,
) -> Dict[str, ECEResult]:
    """
    Compute ECE over records that have a confidence_score and a correct
    label. Records missing either are silently skipped.

    If group_by is given (e.g. `lambda r: r.tool_used.value`, or
    `lambda r: r.meta.get("task_tool_category")` for the cross-tool
    calibration-transfer experiment), returns per-group ECE plus an
    "aggregate" key computed over all eligible records pooled together.

    Returns: {"aggregate": ECEResult, group_key: ECEResult, ...}
    """
    eligible = [r for r in records if r.confidence_score is not None and r.correct is not None]

    result: Dict[str, ECEResult] = {
        "aggregate": _ece_single_group(
            [r.confidence_score for r in eligible],
            [bool(r.correct) for r in eligible],
            n_bins,
        )
    }

    if group_by is not None:
        groups: Dict[str, List[TaskRecord]] = {}
        for r in eligible:
            key = group_by(r)
            groups.setdefault(key, []).append(r)
        for key, group_records in groups.items():
            result[key] = _ece_single_group(
                [r.confidence_score for r in group_records],
                [bool(r.correct) for r in group_records],
                n_bins,
            )

    return result


# ---------------------------------------------------------------------------
# Routing precision / recall / F1
# ---------------------------------------------------------------------------

@dataclass
class RoutingPRResult:
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    tp: int
    fp: int
    fn: int
    tn: int
    n_excluded_ambiguous: int


def compute_routing_precision_recall(
    records: Sequence[TaskRecord], include_ambiguous: bool = False
) -> RoutingPRResult:
    """
    Ground truth: r.tool_necessity == REQUIRED  -> tool call was needed
    Prediction:   r.tool_called == True          -> system called a tool

        TP: needed a tool, system called one
        FP: didn't need a tool, system called one anyway (unnecessary call)
        FN: needed a tool, system didn't call one (missed call)
        TN: didn't need a tool, system didn't call one

    AMBIGUOUS-labeled tasks are excluded by default (their ground truth
    isn't clean enough to grade routing on), and the count excluded is
    reported so it doesn't silently vanish from the analysis.
    """
    tp = fp = fn = tn = 0
    n_excluded = 0

    for r in records:
        if r.tool_necessity == ToolNecessity.AMBIGUOUS and not include_ambiguous:
            n_excluded += 1
            continue

        needed = r.tool_necessity == ToolNecessity.REQUIRED
        called = r.tool_called

        if needed and called:
            tp += 1
        elif not needed and called:
            fp += 1
        elif needed and not called:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    return RoutingPRResult(
        precision=precision, recall=recall, f1=f1,
        tp=tp, fp=fp, fn=fn, tn=tn,
        n_excluded_ambiguous=n_excluded,
    )


def compute_unnecessary_call_rate(records: Sequence[TaskRecord]) -> Optional[float]:
    """
    Of tasks that did NOT require a tool, what fraction got a tool call
    anyway? (FP rate relative to the not-required population — this is
    the more interpretable framing for a cost-focused paper: "how often
    does the router burn money on a tool it didn't need?")
    """
    not_required = [r for r in records if r.tool_necessity == ToolNecessity.NOT_REQUIRED]
    if not not_required:
        return None
    unnecessary = sum(1 for r in not_required if r.tool_called)
    return unnecessary / len(not_required)


def compute_missed_call_rate(records: Sequence[TaskRecord]) -> Optional[float]:
    """
    Of tasks that DID require a tool, what fraction did the system fail
    to call one for? (FN rate relative to the required population — the
    accuracy-risk complement of unnecessary_call_rate's cost-risk.)
    """
    required = [r for r in records if r.tool_necessity == ToolNecessity.REQUIRED]
    if not required:
        return None
    missed = sum(1 for r in required if not r.tool_called)
    return missed / len(required)


# ---------------------------------------------------------------------------
# Latency breakdown
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    mean_ms: float
    median_ms: float
    p95_ms: float
    n: int


@dataclass
class LatencyBreakdownResult:
    overall: LatencyStats
    direct_only: Optional[LatencyStats]
    tool_only: Optional[LatencyStats]
    confidence_stage_only: LatencyStats


def _stats(values: List[float]) -> LatencyStats:
    if not values:
        return LatencyStats(mean_ms=0.0, median_ms=0.0, p95_ms=0.0, n=0)
    values_sorted = sorted(values)
    n = len(values_sorted)
    mean = statistics.mean(values_sorted)
    median = statistics.median(values_sorted)
    # simple nearest-rank p95, no interpolation needed for eval reporting
    p95_idx = min(n - 1, max(0, int(round(0.95 * (n - 1)))))
    p95 = values_sorted[p95_idx]
    return LatencyStats(mean_ms=mean, median_ms=median, p95_ms=p95, n=n)


def compute_latency_breakdown(records: Sequence[TaskRecord]) -> LatencyBreakdownResult:
    all_total = [r.latency.total_ms for r in records]
    direct = [r.latency.total_ms for r in records if r.routing_decision == RoutingDecision.DIRECT]
    tool = [r.latency.total_ms for r in records if r.routing_decision == RoutingDecision.TOOL]
    confidence_only = [r.latency.confidence_ms for r in records]

    return LatencyBreakdownResult(
        overall=_stats(all_total),
        direct_only=_stats(direct) if direct else None,
        tool_only=_stats(tool) if tool else None,
        confidence_stage_only=_stats(confidence_only),
    )
