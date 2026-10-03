"""
analyze.py
==========

Turns a directory of run logs into the three tables the project reports.

    python -m examples.analyze                 # reads runs/
    python -m examples.analyze --log-dir runs --method entropy

report.py summarises one run. Every claim here is comparative, so this
reads them all:

  1. ESTIMATORS   same tasks, four confidence signals, side by side.
     The column that matters is what the estimator itself cost, which is
     the comparison the project exists to make.

  2. TRANSFER     a threshold tuned on one dataset, applied to another.
     The diagonal is each dataset's own best; off-diagonal is what an
     agent with one cut and several tools actually gets. The gap is the
     headline result.

  3. BREAK-EVEN   the per-call tool fee at which routing's spend equals
     always-calling. Spend, not cost-per-success: correctness under a
     path never taken is not in the log, so it cannot be computed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness.analysis import (
    breakeven_tool_fee, compare_runs, load_run_dir, transfer_matrix,
)


def _pct(x):
    return "   n/a" if x is None else f"{x * 100:5.1f}%"


def _usd(x):
    return "      n/a" if x is None else f"${x:.6f}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compare runs across estimators and datasets.")
    p.add_argument("--log-dir", default="runs")
    p.add_argument("--method", default=None,
                   help="restrict the transfer matrix to one estimator "
                        "(default: whichever has the most datasets)")
    args = p.parse_args(argv)

    grouped = load_run_dir(args.log_dir)
    if not grouped:
        print(f"No .jsonl logs in {args.log_dir}/", file=sys.stderr)
        return 1

    # ---- 1. estimator table ----------------------------------------------
    rows = compare_runs(grouped)
    print("=" * 108)
    print("  ESTIMATORS")
    print("=" * 108)
    print(f"{'dataset':11} {'method':17} {'n':>5} {'succ':>7} {'CPST':>10} "
          f"{'ECE':>7} {'P':>7} {'R':>7} {'F1':>7} {'tool%':>7} {'conf $':>10} {'conf%':>7}")
    print("-" * 108)
    for r in rows:
        print(f"{r.dataset:11} {r.method:17} {r.n:>5} {_pct(r.success_rate)} "
              f"{_usd(r.cpst_usd)} "
              f"{'  n/a' if r.ece is None else f'{r.ece:7.4f}'} "
              f"{_pct(r.precision)} {_pct(r.recall)} {_pct(r.f1)} "
              f"{_pct(r.tool_call_rate)} {_usd(r.confidence_cost_usd)} "
              f"{_pct(r.confidence_cost_share)}")

    # ---- 2. transfer matrix ----------------------------------------------
    # One estimator at a time: mixing them would confound a threshold that
    # does not transfer with a confidence signal that is scaled differently.
    by_method = {}
    for (dataset, method), records in grouped.items():
        by_method.setdefault(method, {})[dataset] = records

    method = args.method or max(by_method, key=lambda m: len(by_method[m]))
    datasets = by_method.get(method, {})

    print()
    print("=" * 108)
    print(f"  TRANSFER — threshold tuned on one dataset, applied to another  "
          f"(estimator: {method})")
    print("=" * 108)
    if len(datasets) < 2:
        print(f"  Needs >=2 datasets for '{method}'; found "
              f"{sorted(datasets) or 'none'}.")
        print("  Run the same estimator on a second dataset to get this table.")
    else:
        cells = transfer_matrix(datasets)
        print(f"{'tuned on':12} {'evaluated on':14} {'threshold':>10} "
              f"{'F1':>8} {'own best':>9} {'lost':>8}")
        print("-" * 108)
        for c in cells:
            same = c.tuned_on == c.evaluated_on
            print(f"{c.tuned_on:12} {c.evaluated_on:14} {c.threshold:>10.2f} "
                  f"{_pct(c.f1)} {_pct(c.in_domain_f1)} "
                  f"{'       —' if same else _pct(c.degradation)}")
        worst = max((c for c in cells if c.degradation is not None
                     and c.tuned_on != c.evaluated_on),
                    key=lambda c: c.degradation, default=None)
        if worst is not None:
            print(f"\n  Worst transfer: tuned on {worst.tuned_on}, "
                  f"evaluated on {worst.evaluated_on} — "
                  f"{worst.degradation * 100:.1f} F1 points lost.")

    # ---- 3. break-even ----------------------------------------------------
    print()
    print("=" * 108)
    print("  BREAK-EVEN — per-call tool fee at which routing's spend equals always-calling")
    print("=" * 108)
    print(f"{'dataset':11} {'method':17} {'direct':>7} {'tool':>6} "
          f"{'routed $':>11} {'always $':>11} {'conf $':>10} {'break-even fee':>15}")
    print("-" * 108)
    for (dataset, m) in sorted(grouped):
        b = breakeven_tool_fee(grouped[(dataset, m)])
        fee = ("  n/a (none direct)" if b.breakeven_fee_usd is None
               else f"${b.breakeven_fee_usd:.6f}")
        print(f"{dataset:11} {m:17} {b.direct_calls:>7} {b.tool_calls:>6} "
              f"{b.routed_spend_usd:>11.6f} {b.always_tool_spend_usd:>11.6f} "
              f"{b.confidence_spend_usd:>10.6f} {fee:>15}")
    print()
    print("  A fee at or below zero means the estimator is cheap enough to pay for")
    print("  itself against a free tool. A positive fee is what the tool must charge")
    print("  before routing is worth running at all.")
    print("=" * 108)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
