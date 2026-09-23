"""
run_experiment.py
=================

The entry point for a real run: dataset in, JSONL log and a report out.

    # capability check first -- one cheap call, no sweep
    python -m examples.run_experiment --probe

    # a small real run
    python -m examples.run_experiment --dataset gsm8k --limit 50

    # compare estimators on the same tasks
    python -m examples.run_experiment --dataset gsm8k --limit 50 --method self_consistency

Everything the harness does is already covered by modules that do not
import each other; this script is the wiring, and deliberately nothing
else. No metric is computed here, no routing decision is made here --
if a number needs producing, it comes from report.py.

Costs are counterfactual by design. Runs go through a free tier, and
CPST reports what the same model would have cost at its paid rate. That
is the only way to compare estimators whose entire difference is price.
See costs.py.
"""

from __future__ import annotations

import argparse
import datetime
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness import RunLogger, build_report, print_report
from eval_harness.client import DEFAULT_PROVIDER, Client, MissingAPIKey
from eval_harness.confidence import ConfidenceMethod
from eval_harness.labeling import apply_labels, load_labels
from eval_harness.models import ToolNecessity
from eval_harness.runner import RunConfig, run_dataset
from task_datasets import load_dataset

METHODS = {m.value: m for m in ConfidenceMethod}


def _progress(done: int, total: int, record) -> None:
    mark = "." if record is not None else "!"
    end = "\n" if done == total else ""
    print(mark, end=end, flush=True)
    if done % 50 == 0 and done != total:
        print(f" {done}/{total}", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[3])
    p.add_argument("--dataset", default="gsm8k", help="gsm8k | humaneval | coqa")
    p.add_argument("--limit", type=int, default=50, help="tasks to run (default 50)")
    p.add_argument("--method", default="entropy", choices=sorted(METHODS),
                   help="which confidence estimator to use")
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--k", type=int, default=5, help="self-consistency samples")
    p.add_argument("--provider", default=DEFAULT_PROVIDER)
    p.add_argument("--model", default="gpt-oss-120b")
    p.add_argument("--verifier-model", default="gpt-oss-120b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--log-dir", default="runs")
    p.add_argument("--labels", default=None,
                   help="tool_necessity labels (default labels/<dataset>.json "
                        "if present); produced by examples.label_pilot")
    p.add_argument("--probe", action="store_true",
                   help="check the model returns logprobs, then exit")
    args = p.parse_args(argv)

    try:
        client = Client(provider=args.provider)
    except MissingAPIKey as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2

    # The docs cannot settle whether a given model returns logprobs --
    # Cerebras says support varies by model, OpenRouter's depends on the
    # backing provider. Without them the entropy estimator silently
    # scores None on every task, which is a discovery worth making in
    # one call rather than at the end of a sweep.
    if args.probe:
        result = client.probe(args.model)
        for key in ("provider", "model", "ok", "logprobs", "top_logprobs", "error"):
            print(f"  {key:14} {result.get(key)}")
        if result.get("ok") and not result.get("logprobs"):
            print("\n  This model returned no logprobs. Token entropy will score "
                  "None on every task; use --method self_consistency, or pick "
                  "another model.")
        return 0 if result.get("ok") else 1

    tasks = load_dataset(args.dataset, max_samples=args.limit)
    if not tasks:
        print(f"No tasks loaded for {args.dataset!r}.", file=sys.stderr)
        return 1

    # Labels come from the pilot. Producing them from the model's own
    # failures and then scoring the router on the same tasks would make
    # "needed a tool" and "the router escalated" two readings of one
    # measurement, which is why label_pilot holds half the tasks back.
    label_path = args.labels or f"labels/{args.dataset}.json"
    if Path(label_path).exists():
        label_set = load_labels(label_path)
        if label_set.get("model") and label_set["model"] != args.model:
            warnings.warn(
                f"labels in {label_path} were produced with "
                f"{label_set['model']!r} but this run uses {args.model!r}. "
                f"tool_necessity is model-relative, so these labels describe "
                f"a different model's limits.", UserWarning, stacklevel=1)
        labelled = apply_labels(tasks, label_set, strict=False)
        print(f"labels: {sum(1 for t in labelled if t.get('tool_necessity'))}"
              f"/{len(tasks)} from {label_path}")
        tasks = labelled
    elif args.labels:
        print(f"No label file at {label_path}", file=sys.stderr)
        return 1

    unlabelled = sum(1 for t in tasks if not t.get("tool_necessity"))
    if unlabelled:
        # Not fatal: the cost and calibration numbers are still real. But
        # every routing metric grades against tool_necessity, and unset
        # reads as not_required, so precision/recall would be computed
        # against a ground truth that says no task ever needed a tool.
        warnings.warn(
            f"{unlabelled}/{len(tasks)} tasks have no tool_necessity label, so "
            f"routing precision/recall and the call rates are not meaningful "
            f"for this run. CPST and ECE are unaffected. Label them with a "
            f"pilot run first: python -m examples.label_pilot "
            f"--dataset {args.dataset}", UserWarning, stacklevel=1)

    cfg = RunConfig(
        model=args.model,
        verifier_model=args.verifier_model,
        method=METHODS[args.method],
        threshold=args.threshold,
        k=args.k,
        seed=args.seed,
        run_id=args.run_id or
        f"{args.dataset}_{args.method}_{datetime.datetime.now():%Y%m%d_%H%M%S_%f}",
    )

    print(f"{args.dataset}: {len(tasks)} tasks | {cfg.model} | "
          f"{cfg.method.value} @ {cfg.threshold} | run_id={cfg.run_id}")

    with RunLogger(cfg.run_id, log_dir=args.log_dir) as logger:
        records = run_dataset(client, tasks, cfg, logger=logger, progress=_progress)

    if not records:
        print("No tasks completed.", file=sys.stderr)
        return 1

    print_report(
        build_report(records, ece_group_by=lambda r: r.meta.get("task_tool_category", "none")),
        title=f"{args.dataset} · {cfg.method.value} · {cfg.model}",
    )
    print(f"\nLog: {Path(args.log_dir) / (cfg.run_id + '.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
