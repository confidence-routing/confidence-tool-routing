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
    # Cost is shown in TOKENS: both the estimator and the tool are billed
    # at the same model's rate, so the rate cancels out of every
    # comparison here, and a locally served model has no rate at all.
    print(f"{'dataset':11} {'method':17} {'n':>5} {'succ':>7} {'ECE':>8} "
          f"{'P':>7} {'R':>7} {'F1':>7} {'tool%':>7} "
          f"{'tok/task':>9} {'conf tok':>9} {'conf%':>7} {'CPST tok':>9}")
    print("-" * 108)
    for r in rows:
        per_task = (r.total_tokens / r.n) if r.n else 0
        conf_per_task = (r.confidence_tokens / r.n) if r.n else 0
        print(f"{r.dataset:11} {r.method:17} {r.n:>5} {_pct(r.success_rate)} "
              f"{'     n/a' if r.ece is None else f'{r.ece:8.4f}'} "
              f"{_pct(r.precision)} {_pct(r.recall)} {_pct(r.f1)} "
              f"{_pct(r.tool_call_rate)} "
              f"{per_task:>9.0f} {conf_per_task:>9.0f} "
              f"{_pct(r.confidence_token_share)} "
              f"{'      n/a' if r.cpst_tokens is None else f'{r.cpst_tokens:9.0f}'}")
    if all(r.confidence_cost_usd is None for r in rows):
        print("\n  Dollar columns omitted: this model has no rate on any provider's")
        print("  price page, and estimate_cost_usd() refuses to invent one. Tokens")
        print("  are rate-free and are what every comparison above rests on.")

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
    print("  BREAK-EVEN — did routing cost less than always calling the tool?")
    print("=" * 108)
    print(f"{'dataset':11} {'method':17} {'direct':>7} {'tool':>6} "
          f"{'conf tok':>9} {'tool tok':>9} {'saved':>9} {'spent':>9} "
          f"{'pays off':>9} {'fee needed':>11}")
    print("-" * 108)
    for (dataset, m) in sorted(grouped):
        b = breakeven_tool_fee(grouped[(dataset, m)])
        saved = b.direct_calls * b.mean_tool_tokens
        fee = ("        n/a" if b.breakeven_fee_tokens is None
               else f"{b.breakeven_fee_tokens:11.0f}")
        print(f"{dataset:11} {m:17} {b.direct_calls:>7} {b.tool_calls:>6} "
              f"{b.confidence_tokens:>9} {b.mean_tool_tokens:>9.0f} "
              f"{saved:>9.0f} {b.confidence_tokens:>9} "
              f"{'yes' if b.pays_off_now else 'NO':>9} {fee}")
    print()
    print("  saved = tool tokens the direct answers avoided; spent = tokens the")
    print("  estimator cost across every task. 'fee needed' is the flat per-call")
    print("  charge a tool would have to carry, in token-equivalents, before")
    print("  routing breaks even -- zero means it already pays for itself.")
    print("=" * 108)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
