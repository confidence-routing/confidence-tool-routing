"""
explore_datasets.py
===================

Downloads (on first run) and explores every dataset registered with the
loader framework. For each dataset it prints:

    - Name, tool type, description
    - Total sample count
    - TaskItem field keys
    - Category / label distribution (where applicable)
    - 3 random sample previews (query truncated to 120 chars)

Designed to be run once to sanity-check the loaders before plugging them
into the ground-truth labeling pipeline.

Run:
    python -m examples.explore_datasets                   # all datasets
    python -m examples.explore_datasets gsm8k humaneval   # just those two

For the 210k headlines dataset the first download takes a while (~15 min
via the rows API). Pass --max-samples 1000 to cap download + parse for a
quick sanity check.
"""

from __future__ import annotations

import argparse
import random
import sys
import textwrap
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset, list_datasets, TaskItem


def _trunc(text: str, width: int = 120) -> str:
    text = " ".join(text.split())
    return text[:width] + "..." if len(text) > width else text


def _print_distribution(items: list[TaskItem], field: str, top_n: int = 10) -> None:
    values = [item.get("meta", {}).get(field) or item.get("gold_answer", "") for item in items]
    counter = Counter(values)
    total = len(values)
    print(f"\n  Distribution ({field}, top {top_n}):")
    for value, count in counter.most_common(top_n):
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"    {value:30s}  {count:6d}  ({pct:5.1f}%)  {bar}")
    if len(counter) > top_n:
        print(f"    ... and {len(counter) - top_n} more unique values")


def explore_one(name: str, max_samples: int | None) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  DATASET: {name}")
    print(bar)

    try:
        items = load_dataset(name, max_samples=max_samples)
    except Exception as e:
        print(f"  ERROR loading '{name}': {e}")
        return

    if not items:
        print("  (empty)")
        return

    sample = items[0]
    print(f"  Tool type:     {sample.get('tool_type', '?')}")
    print(f"  Total samples: {len(items)}")
    print(f"  TaskItem keys: {sorted(sample.keys())}")

    # Meta keys
    meta_keys = sorted(set().union(*(item.get("meta", {}).keys() for item in items[:100])))
    print(f"  Meta keys:     {meta_keys}")

    # Distribution — pick the most informative field per dataset
    if name == "headlines":
        _print_distribution(items, "category")
    elif name == "overruling":
        _print_distribution(items, "label")
    elif name == "gsm8k":
        # Show answer magnitude distribution
        def _bucket(item):
            try:
                v = float(item["gold_answer"].replace(",", ""))
                if v < 10: return "<10"
                if v < 100: return "10-99"
                if v < 1000: return "100-999"
                if v < 10000: return "1k-9.9k"
                return "10k+"
            except (ValueError, KeyError):
                return "non-numeric"
        counter = Counter(_bucket(item) for item in items)
        print(f"\n  Answer magnitude distribution:")
        for bucket in ["<10", "10-99", "100-999", "1k-9.9k", "10k+", "non-numeric"]:
            count = counter.get(bucket, 0)
            pct = count / len(items) * 100
            print(f"    {bucket:15s}  {count:6d}  ({pct:5.1f}%)")
    elif name == "coqa":
        # Show domain distribution
        domains = Counter(item.get("meta", {}).get("source_domain", "?") for item in items)
        print(f"\n  Domain distribution:")
        for domain, count in domains.most_common():
            pct = count / len(items) * 100
            print(f"    {domain:20s}  {count:6d}  ({pct:5.1f}%)")
        # Turns per conversation
        stories = Counter(item.get("meta", {}).get("story_id") for item in items)
        turns = list(stories.values())
        print(f"\n  Conversations: {len(stories)}")
        print(f"  Turns per conversation: min={min(turns)}, max={max(turns)}, "
              f"mean={sum(turns)/len(turns):.1f}")
    elif name == "humaneval":
        print(f"\n  All 164 tasks are unique Python functions.")

    # Random samples
    rng = random.Random(42)
    sample_items = rng.sample(items, min(3, len(items)))
    print(f"\n  --- Sample TaskItems ({min(3, len(items))}) ---")
    for i, item in enumerate(sample_items):
        print(f"\n  [{i+1}] task_id: {item['task_id']}")
        print(f"      query:      {_trunc(item['query'])}")
        print(f"      gold:       {_trunc(item['gold_answer'], 80)}")
        print(f"      tool_type:  {item['tool_type']}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore registered datasets")
    parser.add_argument("names", nargs="*", help="Dataset names (default: all)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap samples per dataset (useful for quick checks)")
    args = parser.parse_args()

    all_datasets = list_datasets()

    if args.names:
        targets = args.names
    else:
        targets = [d["name"] for d in all_datasets]

    print("Registered datasets:")
    for d in all_datasets:
        marker = " *" if d["name"] in targets else ""
        print(f"  {d['name']:15s}  [{d['tool_type']:14s}]  {d['description']}{marker}")

    for name in targets:
        explore_one(name, args.max_samples)

    print("Done.")


if __name__ == "__main__":
    main()
