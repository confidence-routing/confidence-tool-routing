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

BOTH halves are labelled, and each is tagged with which split it belongs
to. An earlier version labelled only the pilot half, which left the
evaluation half with no ground truth and therefore no routing metrics at
all -- precision, recall and both call rates key off tool_necessity.

The split still matters, just later: the threshold is selected on the
pilot half and reported on the eval half, so the cut is never tuned on
the sample it is scored against. That guard belongs at threshold
selection, not at labelling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness.client import DEFAULT_PROVIDER, Client, MissingAPIKey
from eval_harness.labeling import (
    label_tool_necessity, load_labels, merge_label_sets, save_labels,
    split_tasks, summarize,
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
    p.add_argument("--split", default="all",
                   choices=["all", "pilot", "eval"],
                   help="which half to label (default all); labels merge "
                        "into any existing file rather than replacing it")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    try:
        client = Client(provider=args.provider)
    except MissingAPIKey as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2

    tasks = load_dataset(args.dataset, max_samples=args.limit)
    split = split_tasks(tasks, pilot_fraction=args.pilot_fraction, seed=args.seed)
    wanted = ["pilot", "eval"] if args.split == "all" else [args.split]

    print(f"{args.dataset}: {len(tasks)} loaded -> {len(split['pilot'])} pilot / "
          f"{len(split['eval'])} eval | labelling: {', '.join(wanted)} | "
          f"{args.model} | {args.trials} trials")
    print("  R = needed a tool, . = did not, ? = inconsistent")

    out = args.out or f"labels/{args.dataset}.json"
    # Merge rather than replace: each half costs real time to label, and
    # adding the second must not discard the first.
    label_set = load_labels(out) if Path(out).exists() else {}

    for name in wanted:
        print(f"  [{name}]", end=" ", flush=True)
        part = label_tool_necessity(
            client, split[name], args.model, trials=args.trials,
            temperature=args.temperature, split=name, progress=_progress)
        label_set = merge_label_sets(label_set, part)
        save_labels(label_set, out)   # checkpoint after each half

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
