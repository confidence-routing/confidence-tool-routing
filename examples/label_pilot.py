"""
label_pilot.py
==============

Runs the pilot that produces ``tool_necessity`` labels.

    python -m examples.label_pilot --dataset gsm8k --limit 100

Attempts each task with no tool, several times, and labels it by how
often the model got it right: always -> not_required, never -> required,
sometimes -> ambiguous. Labels are written to labels/<dataset>.json and
picked up by run_experiment.

Run this before any routing number is worth reading. Without labels,
unset reads as not_required downstream, so precision and recall get
graded against a ground truth claiming no task ever needed a tool.

It also answers a question we have so far only assumed: whether a
dataset is ceilinged. If the model solves nearly everything unaided,
almost every task labels not_required, the positive class is empty, and
routing cannot be measured on it whatever the estimator does. The
summary prints that split, so it is a number rather than a hunch.

Only the PILOT half is labelled. The eval half is held out on purpose:
scoring the router on the same tasks whose labels came from the model's
own failures would make "needed a tool" and "the router escalated" two
readings of one measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness.client import DEFAULT_PROVIDER, Client, MissingAPIKey
from eval_harness.labeling import (
    label_tool_necessity, save_labels, split_tasks, summarize,
)
from task_datasets import load_dataset


def _progress(done: int, total: int, label: str) -> None:
    print({"required": "R", "not_required": ".", "ambiguous": "?"}.get(label, "-"),
          end="\n" if done == total else "", flush=True)
    if done % 50 == 0 and done != total:
        print(f" {done}/{total}", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Label tool_necessity from a pilot run.")
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--limit", type=int, default=100, help="tasks to load before splitting")
    p.add_argument("--trials", type=int, default=3, help="attempts per task")
    p.add_argument("--pilot-fraction", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--provider", default=DEFAULT_PROVIDER)
    p.add_argument("--model", default="gpt-oss-120b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    try:
        client = Client(provider=args.provider)
    except MissingAPIKey as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2

    tasks = load_dataset(args.dataset, max_samples=args.limit)
    split = split_tasks(tasks, pilot_fraction=args.pilot_fraction, seed=args.seed)
    pilot = split["pilot"]

    print(f"{args.dataset}: {len(tasks)} loaded -> {len(pilot)} pilot / "
          f"{len(split['eval'])} held out | {args.model} | {args.trials} trials")
    print("  R = needed a tool, . = did not, ? = inconsistent")

    label_set = label_tool_necessity(
        client, pilot, args.model, trials=args.trials,
        temperature=args.temperature, progress=_progress)

    out = args.out or f"labels/{args.dataset}.json"
    save_labels(label_set, out)

    counts = summarize(label_set)
    total = sum(counts.values()) or 1
    print(f"\n  required      {counts['required']:>4}  ({counts['required']/total*100:.0f}%)")
    print(f"  not_required  {counts['not_required']:>4}  ({counts['not_required']/total*100:.0f}%)")
    print(f"  ambiguous     {counts['ambiguous']:>4}  ({counts['ambiguous']/total*100:.0f}%)")
    print(f"\n  -> {out}")

    gradeable = counts["required"] + counts["not_required"]
    if gradeable and counts["required"] / gradeable < 0.10:
        print(f"\n  WARNING: only {counts['required']} of {gradeable} gradeable tasks "
              f"needed a tool. This dataset is close to ceilinged for this model -- "
              f"routing recall is computed over that tiny positive class, so it will "
              f"be extremely noisy. Consider a harder dataset or a smaller model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
