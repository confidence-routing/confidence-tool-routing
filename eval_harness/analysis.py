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
    ece: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    unnecessary_call_rate: Optional[float]
    missed_call_rate: Optional[float]
    tool_call_rate: float
    confidence_cost_usd: float      # what the estimator itself cost
    confidence_cost_share: float    # as a fraction of total spend


def summarize_run(dataset: str, method: str, records: Sequence[TaskRecord]) -> RunSummary:
    """Headline numbers for one (dataset, method), as it actually ran."""
    cpst = compute_cpst(records)
    ece = compute_ece(records)["aggregate"]
    routing = compute_routing_precision_recall(records)
    n = len(records)

    # What the confidence stage cost, separated out. This is the column the
    # whole project turns on: without it, free token entropy and k-sample
    # self-consistency are indistinguishable in a results table.
    confidence_usd = 0.0
    for r in records:
        full = estimate_record_cost_usd(r)
        stripped = estimate_record_cost_usd(_without_confidence_stage(r))
        confidence_usd += full - stripped

    return RunSummary(
        dataset=dataset, method=method, n=n,
        success_rate=cpst.success_rate,
        cpst_usd=cpst.cpst_usd,
        ece=ece.ece,
        precision=routing.precision, recall=routing.recall, f1=routing.f1,
        unnecessary_call_rate=None, missed_call_rate=None,
        tool_call_rate=(sum(1 for r in records if r.tool_called) / n) if n else 0.0,
        confidence_cost_usd=confidence_usd,
        confidence_cost_share=(confidence_usd / cpst.total_cost_usd
                               if cpst.total_cost_usd else 0.0),
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
    routed_spend_usd: float          # what the run actually spent
    always_tool_spend_usd: float     # if every task had called the tool
    tool_calls: int
    direct_calls: int
    confidence_spend_usd: float
    breakeven_fee_usd: Optional[float]
    current_fee_usd: float
    pays_off_now: bool


def breakeven_tool_fee(records: Sequence[TaskRecord]) -> BreakevenResult:
    """
    The per-call tool fee at which routing's SPEND equals always-calling.

    This is spend, not cost per successful task, and the distinction is
    the honest part. Cost per success needs correctness under the
    counterfactual, which is not in the log. Spend needs only token
    counts and decisions, which are.

    The arithmetic the project rests on: routing saves the tool cost of
    every task it answered directly, and pays the estimator's cost on
    every task either way. So it wins only when

        direct_calls * (tool token cost + fee) > estimator cost on all tasks

    which solves for a fee. Below it, the estimator costs more than the
    tool calls it prevented -- and for a free tool like a calculator there
    is no fee to clear, so only an estimator that is itself free can win.
    That is arithmetic rather than a finding, and it is why the fee a tool
    carries is the variable worth reporting against.

    Returns breakeven_fee_usd = None when no task was answered directly
    (nothing was saved, so no fee makes routing cheaper).
    """
    routed = sum(estimate_record_cost_usd(r) for r in records)
    confidence = sum(estimate_record_cost_usd(r)
                     - estimate_record_cost_usd(_without_confidence_stage(r))
                     for r in records)

    tool_records = [r for r in records if r.tool_called]
    direct = [r for r in records if not r.tool_called]
    n_direct = len(direct)

    # Mean observed tool round-trip, in tokens-as-dollars plus whatever
    # flat fee was logged, taken from the tasks that actually called it.
    if tool_records:
        tool_token_cost = sum(
            estimate_record_cost_usd(r) - estimate_record_cost_usd(_without_tool_stage(r))
            for r in tool_records) / len(tool_records)
        current_fee = sum(r.tool_api_cost_usd for r in tool_records) / len(tool_records)
    else:
        tool_token_cost, current_fee = 0.0, 0.0

    always_tool = routed + n_direct * tool_token_cost

    breakeven = None
    if n_direct:
        # confidence_spend = n_direct * (tool_token_cost_without_fee + fee)
        tool_tokens_only = max(tool_token_cost - current_fee, 0.0)
        breakeven = (confidence / n_direct) - tool_tokens_only

    return BreakevenResult(
        routed_spend_usd=routed,
        always_tool_spend_usd=always_tool,
        tool_calls=len(tool_records),
        direct_calls=n_direct,
        confidence_spend_usd=confidence,
        breakeven_fee_usd=breakeven,
        current_fee_usd=current_fee,
        pays_off_now=routed < always_tool,
    )


def _without_tool_stage(record: TaskRecord) -> TaskRecord:
    from dataclasses import replace
    return replace(record, tool_prompt_tokens=0, tool_completion_tokens=0,
                   tool_api_cost_usd=0.0)
