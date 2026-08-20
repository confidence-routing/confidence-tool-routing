"""
report.py
=========

Turns a list of TaskRecord into a single summary dict (build_report) and
a human-readable printout (print_report). This is what you'd call at the
end of an experiment run, or point at a saved .jsonl log for offline
analysis.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from .models import TaskRecord
from .metrics import (
    compute_cpst,
    compute_ece,
    compute_routing_precision_recall,
    compute_unnecessary_call_rate,
    compute_missed_call_rate,
    compute_latency_breakdown,
)


def build_report(
    records: Sequence[TaskRecord],
    ece_group_by: Optional[Callable[[TaskRecord], str]] = None,
    ece_bins: int = 10,
) -> Dict:
    """Compute every metric and pack it into one dict — the thing you'd
    dump to JSON for a paper table, or feed into a plotting script."""
    cpst = compute_cpst(records)
    ece = compute_ece(records, n_bins=ece_bins, group_by=ece_group_by)
    routing = compute_routing_precision_recall(records)
    unnecessary_rate = compute_unnecessary_call_rate(records)
    missed_rate = compute_missed_call_rate(records)
    latency = compute_latency_breakdown(records)

    return {
        "n_tasks": len(records),
        "cpst": cpst,
        "ece": ece,
        "routing_precision_recall": routing,
        "unnecessary_call_rate": unnecessary_rate,
        "missed_call_rate": missed_rate,
        "latency": latency,
    }


def _fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _fmt_usd(x: Optional[float]) -> str:
    return "n/a" if x is None else f"${x:.4f}"


def print_report(report: Dict, title: str = "Run Report") -> None:
    cpst = report["cpst"]
    ece = report["ece"]
    routing = report["routing_precision_recall"]
    latency = report["latency"]

    bar = "=" * max(28, len(title) + 4)
    print(bar)
    print(f"  {title}")
    print(bar)
    print(f"Tasks:              {report['n_tasks']}")
    print(f"Success rate:       {_fmt_pct(cpst.success_rate)}  ({cpst.num_correct}/{cpst.num_tasks})")
    print()
    print("-- Cost Per Successful Task --")
    print(f"CPST (USD):         {_fmt_usd(cpst.cpst_usd)}")
    print(f"CPST (tokens):      {'n/a' if cpst.cpst_tokens is None else f'{cpst.cpst_tokens:.1f}'}")
    print(f"Total cost:         {_fmt_usd(cpst.total_cost_usd)}")
    print(f"Total tokens:       {cpst.total_tokens}")
    print()
    print("-- Calibration (ECE) --")
    agg = ece["aggregate"]
    print(f"Aggregate ECE:      {'n/a' if agg.ece is None else f'{agg.ece:.4f}'}  (n={agg.n})")
    for key, res in ece.items():
        if key == "aggregate":
            continue
        print(f"  ECE[{key}]:{' ' * max(1, 12 - len(key))}{'n/a' if res.ece is None else f'{res.ece:.4f}'}  (n={res.n})")
    print()
    print("-- Routing Decision Quality --")
    print(f"Precision:          {_fmt_pct(routing.precision)}")
    print(f"Recall:             {_fmt_pct(routing.recall)}")
    print(f"F1:                 {_fmt_pct(routing.f1)}")
    print(f"TP / FP / FN / TN:  {routing.tp} / {routing.fp} / {routing.fn} / {routing.tn}")
    if routing.n_excluded_ambiguous:
        print(f"(excluded {routing.n_excluded_ambiguous} ambiguous-labeled tasks)")
    print(f"Unnecessary-call rate: {_fmt_pct(report['unnecessary_call_rate'])}")
    print(f"Missed-call rate:      {_fmt_pct(report['missed_call_rate'])}")
    print()
    print("-- Latency (ms) --")
    print(f"Overall:            mean={latency.overall.mean_ms:.0f}  median={latency.overall.median_ms:.0f}  p95={latency.overall.p95_ms:.0f}  (n={latency.overall.n})")
    if latency.direct_only:
        d = latency.direct_only
        print(f"Direct-path:        mean={d.mean_ms:.0f}  median={d.median_ms:.0f}  p95={d.p95_ms:.0f}  (n={d.n})")
    if latency.tool_only:
        t = latency.tool_only
        print(f"Tool-path:          mean={t.mean_ms:.0f}  median={t.median_ms:.0f}  p95={t.p95_ms:.0f}  (n={t.n})")
    c = latency.confidence_stage_only
    print(f"Confidence stage:   mean={c.mean_ms:.0f}  median={c.median_ms:.0f}  p95={c.p95_ms:.0f}  (n={c.n})")
    print(bar)
