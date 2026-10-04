"""
analysis.py
===========

Comparisons across runs. report.py summarises ONE run; the claims in this
project are all comparative, so this is where logs become results.

Three things it does, and the boundary between them matters:

    compare_runs        the estimator table -- same tasks, four confidence
                        signals, side by side
    transfer_matrix     THE headline. Pick a threshold on one dataset,
                        apply it to another, measure what it costs you
    breakeven_tool_fee  the tool price at which routing starts paying for
                        itself

What is comparable across runs and what is not
----------------------------------------------
Routing quality survives counterfactual recomputation: precision, recall
and the call rates are functions of ``tool_necessity`` and the decision,
both of which are in the log. So a threshold can be re-applied to a
finished run and graded honestly, which is what makes the transfer matrix
possible without re-running anything.

Correctness does not survive it. A task answered directly and graded
wrong has no recorded answer for the tool path it never took, so CPST
under a counterfactual threshold is not recoverable -- it would be
invented. Every function here that re-decides routing therefore reports
routing metrics only, and the ones that report cost use the decisions as
they actually happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .costs import estimate_record_cost_usd
from .logger import RunLogger
from .metrics import compute_cpst, compute_ece, compute_routing_precision_recall
from .models import RoutingDecision, TaskRecord, ToolNecessity
from .router import DEFAULT_THRESHOLDS, select_threshold, sweep_thresholds

RunKey = Tuple[str, str]       # (dataset, confidence_method)


def _cost_usd(record: TaskRecord) -> Optional[float]:
    """Dollar cost, or None when the model has no rate.

    A locally served model has no price, and estimate_cost_usd() rightly
    refuses to invent one. None means "no rate", never "free".
    """
    try:
        return estimate_record_cost_usd(record)
    except KeyError:
        return None


def _confidence_tokens(record: TaskRecord) -> int:
    """Tokens the confidence stage spent. Rate-free, so this is the column
    that still works for an unpriced model -- and the one that makes the
    estimator comparison possible at all."""
    return record.confidence_prompt_tokens + record.confidence_completion_tokens


def _tool_tokens(record: TaskRecord) -> int:
    return record.tool_prompt_tokens + record.tool_completion_tokens


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_runs(paths: Iterable[str]) -> Dict[RunKey, List[TaskRecord]]:
    """
    Read several JSONL logs and group their records by (dataset, method).

    Grouped from the records rather than from filenames: a run_id is a
    convention and a filename is an accident, while dataset and
    confidence_method are fields the pipeline actually set. A log holding
    two methods splits correctly instead of being mislabelled by its name.
    """
    grouped: Dict[RunKey, List[TaskRecord]] = {}
    for path in paths:
        for record in RunLogger.iter_load(str(path)):
            key = (record.dataset, record.confidence_method.value)
            grouped.setdefault(key, []).append(record)
    return grouped


def load_run_dir(log_dir: str = "runs") -> Dict[RunKey, List[TaskRecord]]:
    """Every .jsonl in a directory, grouped as load_runs does."""
    return load_runs(sorted(str(p) for p in Path(log_dir).glob("*.jsonl")))


# ---------------------------------------------------------------------------
# The estimator table
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    dataset: str
    method: str
    n: int
    success_rate: float
    cpst_usd: Optional[float]
    cpst_tokens: Optional[float]
    ece: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    unnecessary_call_rate: Optional[float]
    missed_call_rate: Optional[float]
    tool_call_rate: float
    # What the estimator itself cost. USD is None for an unpriced model;
    # the token figures always work and are what the comparison rests on.
    confidence_cost_usd: Optional[float]
    confidence_cost_share: Optional[float]
    confidence_tokens: int
    confidence_token_share: float
    total_tokens: int


def summarize_run(dataset: str, method: str, records: Sequence[TaskRecord]) -> RunSummary:
    """Headline numbers for one (dataset, method), as it actually ran."""
    cpst = compute_cpst(records)
    ece = compute_ece(records)["aggregate"]
    routing = compute_routing_precision_recall(records)
    n = len(records)

    # What the confidence stage cost, separated out. This is the column the
    # whole project turns on: without it, free token entropy and k-sample
    # self-consistency are indistinguishable in a results table.
    confidence_usd: Optional[float] = 0.0
    confidence_tokens = 0
    total_tokens = 0
    for r in records:
        confidence_tokens += _confidence_tokens(r)
        total_tokens += r.total_prompt_tokens + r.total_completion_tokens
        if confidence_usd is not None:
            full = _cost_usd(r)
            stripped = _cost_usd(_without_confidence_stage(r))
            if full is None or stripped is None:
                confidence_usd = None   # unpriced: stop, do not report a partial total
            else:
                confidence_usd += full - stripped

    return RunSummary(
        dataset=dataset, method=method, n=n,
        success_rate=cpst.success_rate,
        cpst_usd=cpst.cpst_usd,
        cpst_tokens=cpst.cpst_tokens,
        ece=ece.ece,
        precision=routing.precision, recall=routing.recall, f1=routing.f1,
        unnecessary_call_rate=None, missed_call_rate=None,
        tool_call_rate=(sum(1 for r in records if r.tool_called) / n) if n else 0.0,
        confidence_cost_usd=confidence_usd,
        confidence_cost_share=(confidence_usd / cpst.total_cost_usd
                               if (confidence_usd is not None and cpst.total_cost_usd)
                               else None),
        confidence_tokens=confidence_tokens,
        confidence_token_share=(confidence_tokens / total_tokens
                                if total_tokens else 0.0),
        total_tokens=total_tokens,
    )


def _without_confidence_stage(record: TaskRecord) -> TaskRecord:
    """A copy with the confidence-stage costs zeroed, for differencing."""
    from dataclasses import replace
    return replace(record, confidence_prompt_tokens=0,
                   confidence_completion_tokens=0, confidence_api_cost_usd=0.0)


def compare_runs(grouped: Dict[RunKey, List[TaskRecord]]) -> List[RunSummary]:
    """One RunSummary per (dataset, method), dataset-major then method."""
    from .metrics import compute_missed_call_rate, compute_unnecessary_call_rate

    out: List[RunSummary] = []
    for (dataset, method) in sorted(grouped):
        records = grouped[(dataset, method)]
        summary = summarize_run(dataset, method, records)
        summary.unnecessary_call_rate = compute_unnecessary_call_rate(records)
        summary.missed_call_rate = compute_missed_call_rate(records)
        out.append(summary)
    return out


# ---------------------------------------------------------------------------
# Objectives for threshold selection
# ---------------------------------------------------------------------------
# Which objective you pick decides what the transfer matrix measures, and
# "f1" is the wrong default on this data. F1 rewards predicting the
# majority class, and CoQA is 72% tool-required -- so maximising it pushes
# the threshold to 1.00, i.e. "escalate always". Two datasets then agree
# on a degenerate cut and transfer looks perfect while measuring nothing.
#
# cost_weighted asks the question a deployment actually asks: among cuts
# that miss no more than max_missed of the tasks that needed a tool, take
# the one that wastes the fewest calls on tasks that did not. That has a
# real operating point, so a threshold selected under it carries
# information a second dataset can disagree with.


def cost_weighted_objective(max_missed: float = 0.10) -> Callable:
    """Fewest unnecessary calls, subject to a cap on missed ones.

    Returns None for a point that breaks the constraint, which
    select_threshold skips -- so if no cut satisfies it, selection
    reports nothing rather than quietly relaxing the requirement.
    """
    def objective(point) -> Optional[float]:
        if point.missed_call_rate is None or point.unnecessary_call_rate is None:
            return None
        if point.missed_call_rate > max_missed:
            return None
        return -point.unnecessary_call_rate
    return objective


OBJECTIVES: Dict[str, Callable] = {
    "f1": lambda p: p.f1,
    "cost": cost_weighted_objective(0.10),
    "cost25": cost_weighted_objective(0.25),
}


# ---------------------------------------------------------------------------
# Cross-tool transfer -- the headline
# ---------------------------------------------------------------------------

@dataclass
class TransferCell:
    tuned_on: str
    evaluated_on: str
    threshold: float
    f1: Optional[float]
    in_domain_f1: Optional[float]   # best achievable on evaluated_on
    degradation: Optional[float]    # in_domain_f1 - f1, >0 means transfer hurt


def transfer_matrix(
    records_by_dataset: Dict[str, Sequence[TaskRecord]],
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    objective: str | Callable = "f1",
    on_missing: str = "tool",
) -> List[TransferCell]:
    """
    Tune a threshold on each dataset, then apply it to every dataset.

    The diagonal is the best each dataset can do with its own threshold.
    Off-diagonal is what you get when the threshold came from somewhere
    else, which is the situation any real agent is in: it has one cut and
    several kinds of tool.

    ``degradation`` is the gap, and it is the result. A small gap says a
    threshold is a property of the model; a large one says it is a
    property of the task, and that tuning on one tool category tells you
    little about another.

    Only routing metrics appear here, deliberately. Re-deciding routing
    over a finished log is sound -- tool_necessity and the decision are
    both logged -- but correctness under a path the system never took is
    not, so there is no CPST column to report and inventing one would be
    the easiest mistake in this file to make.
    """
    tuned: Dict[str, float] = {}
    in_domain: Dict[str, Optional[float]] = {}
    for dataset, records in records_by_dataset.items():
        best = select_threshold(records, thresholds, objective=objective,
                                on_missing=on_missing)
        if best is None:
            continue
        tuned[dataset] = best.threshold
        in_domain[dataset] = best.f1

    cells: List[TransferCell] = []
    for source, threshold in sorted(tuned.items()):
        for target in sorted(records_by_dataset):
            point = _point_at(records_by_dataset[target], threshold,
                              thresholds, on_missing)
            f1 = point.f1 if point else None
            best = in_domain.get(target)
            cells.append(TransferCell(
                tuned_on=source, evaluated_on=target, threshold=threshold,
                f1=f1, in_domain_f1=best,
                degradation=(best - f1) if (best is not None and f1 is not None) else None,
            ))
    return cells


def _point_at(records, threshold, thresholds, on_missing):
    """The sweep row for one threshold, computed by the same code a real
    run at that threshold would go through."""
    for point in sweep_thresholds(records, [threshold], on_missing=on_missing):
        return point
    return None


# ---------------------------------------------------------------------------
# Break-even on tool price
# ---------------------------------------------------------------------------

@dataclass
class BreakevenResult:
    """
    Whether routing paid for itself, and what it would take.

    Stated in TOKENS as the primary unit. That is not a fallback for an
    unpriced model -- it is the better formulation. Both sides of the
    comparison are billed at the same model's rate, so the rate cancels
    and the break-even is a pure token ratio. Dollar figures are filled
    in when a rate happens to exist, and left None when it does not.
    """
    direct_calls: int
    tool_calls: int
    confidence_tokens: int
    mean_tool_tokens: float
    # Extra tool tokens a direct answer avoided, against what the
    # estimator cost to decide that. >1 means routing paid off.
    saved_per_direct: float
    spent_per_task: float
    pays_off_now: bool
    # The flat per-call fee a tool would need to carry for routing to
    # break even, expressed in token-equivalents of the same model.
    breakeven_fee_tokens: Optional[float]
    routed_spend_usd: Optional[float]
    always_tool_spend_usd: Optional[float]


def breakeven_tool_fee(records: Sequence[TaskRecord]) -> BreakevenResult:
    """
    Did routing cost less than always calling the tool, and if not, how
    much dearer would the tool have to be?

    The arithmetic the project rests on: routing avoids the tool cost of
    every task it answered directly, and pays the estimator on every task
    either way. So it wins when

        direct_calls * tool_cost  >  confidence_cost over ALL tasks

    Both sides are the same model's tokens, so this is decidable without
    any price at all -- which is why tokens are the unit here. A free tool
    (a calculator, a local code runner) costs only its round-trip tokens,
    so against it only an estimator that is itself nearly free can win.
    That is arithmetic rather than a finding, and it is why the fee a tool
    carries is the variable worth reporting against.

    breakeven_fee_tokens is None when nothing was answered directly:
    routing bought nothing, so no fee makes it cheaper.
    """
    n = len(records)
    tool_records = [r for r in records if r.tool_called]
    n_direct = n - len(tool_records)

    confidence_tokens = sum(_confidence_tokens(r) for r in records)
    mean_tool_tokens = (sum(_tool_tokens(r) for r in tool_records) / len(tool_records)
                        if tool_records else 0.0)

    saved = n_direct * mean_tool_tokens
    spent = confidence_tokens

    breakeven_fee = None
    if n_direct:
        # fee needed, in token-equivalents, so that saved >= spent
        breakeven_fee = max((spent - saved) / n_direct, 0.0)

    routed_usd = total = 0.0
    priced = True
    for r in records:
        c = _cost_usd(r)
        if c is None:
            priced = False
            break
        routed_usd += c
    always_usd = None
    if priced and tool_records:
        mean_tool_usd = 0.0
        for r in tool_records:
            full, no_tool = _cost_usd(r), _cost_usd(_without_tool_stage(r))
            if full is None or no_tool is None:
                priced = False
                break
            mean_tool_usd += full - no_tool
        if priced:
            mean_tool_usd /= len(tool_records)
            always_usd = routed_usd + n_direct * mean_tool_usd

    return BreakevenResult(
        direct_calls=n_direct,
        tool_calls=len(tool_records),
        confidence_tokens=confidence_tokens,
        mean_tool_tokens=mean_tool_tokens,
        saved_per_direct=mean_tool_tokens,
        spent_per_task=(spent / n) if n else 0.0,
        pays_off_now=saved >= spent,
        breakeven_fee_tokens=breakeven_fee,
        routed_spend_usd=routed_usd if priced else None,
        always_tool_spend_usd=always_usd,
    )


def _without_tool_stage(record: TaskRecord) -> TaskRecord:
    from dataclasses import replace
    return replace(record, tool_prompt_tokens=0, tool_completion_tokens=0,
                   tool_api_cost_usd=0.0)
