"""
ollama_label_pilot.py
=====================

Produces tool_necessity labels using a local Ollama model. No API key,
no cost, runs entirely offline.

    # install Ollama first: https://ollama.com
    ollama pull llama3.2:3b        # or any model you like

    python -m examples.ollama_label_pilot --dataset gsm8k --limit 50
    python -m examples.ollama_label_pilot --dataset coqa --limit 100 --model llama3.1:8b
    python -m examples.ollama_label_pilot --dataset overruling --limit 50 --trials 5

For each task it asks the model the question N times (no tool, no
passage, no calculator) and labels by how often it got the right answer:

    always right   -> not_required  (the model knows this)
    never right    -> required      (the model needs a tool)
    sometimes      -> ambiguous     (excluded from routing metrics)

Labels are saved to labels/<dataset>.json and used by the experiment
runner to compute routing precision/recall. Without them those metrics
are meaningless.

The output also tells you if a dataset is ceilinged — if the model
solves nearly everything, almost every task is not_required, the positive
class is empty, and routing can't be measured. That's useful information
before committing to a full experiment.

Requires: ollama running locally (default http://localhost:11434)
          pip install openai   (Ollama speaks the OpenAI-compatible API)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness.grading import grade
from eval_harness.models import ToolNecessity


# ---------------------------------------------------------------------------
# Ollama client (minimal, OpenAI-compatible)
# ---------------------------------------------------------------------------

def _make_client(base_url: str = "http://localhost:11434/v1"):
    """Create an OpenAI-compatible client pointed at Ollama."""
    try:
        from openai import OpenAI
    except ImportError:
        print("The 'openai' package is required. Install with:\n"
              "  pip install openai", file=sys.stderr)
        sys.exit(1)
    return OpenAI(base_url=base_url, api_key="ollama")


def _complete(client, model: str, messages: list, **kwargs) -> str:
    """One chat completion, return the text."""
    response = client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Prompt construction (matches the integration branch's convention)
# ---------------------------------------------------------------------------

_ANSWER_SYSTEM = (
    "Answer the question. Be brief. State your final answer on the last "
    "line, with no explanation after it."
)

_COQA_SYSTEM = (
    "Answer the question in as few words as possible, using the wording of "
    "the source where you can. Do not explain."
)


def build_messages(task: dict, *, include_passage: bool = False) -> list:
    """Build the prompt for one task. No tool context by default."""
    dataset = task.get("dataset", "")
    meta = task.get("meta", {}) or {}

    if dataset != "coqa":
        return [{"role": "system", "content": _ANSWER_SYSTEM},
                {"role": "user", "content": task["query"]}]

    messages = [{"role": "system", "content": _COQA_SYSTEM}]
    if include_passage and meta.get("passage"):
        messages.append({"role": "user",
                         "content": f"Passage:\n\n{meta['passage']}"})
    for turn in meta.get("conversation_history") or []:
        messages.append({"role": "user", "content": turn.get("question", "")})
        messages.append({"role": "assistant", "content": turn.get("answer", "")})
    messages.append({"role": "user", "content": task["query"]})
    return messages


# ---------------------------------------------------------------------------
# Labeling logic
# ---------------------------------------------------------------------------

def decide_label(n_correct: int, trials: int) -> str:
    if n_correct >= trials:
        return ToolNecessity.NOT_REQUIRED.value
    if n_correct <= 0:
        return ToolNecessity.REQUIRED.value
    return ToolNecessity.AMBIGUOUS.value


def label_tasks(client, tasks, model, trials, temperature, max_tokens, progress_fn):
    """Label each task by attempting it `trials` times with no tool."""
    labels = {}
    detail = {}

    for i, task in enumerate(tasks):
        task_id = str(task.get("task_id", i))
        messages = build_messages(task)
        n_correct = 0

        for _ in range(trials):
            answer = _complete(client, model, messages,
                               temperature=temperature, max_tokens=max_tokens)
            try:
                if grade(task["dataset"], answer,
                         gold_answer=task.get("gold_answer"),
                         meta=task.get("meta")):
                    n_correct += 1
            except KeyError:
                raise

        label = decide_label(n_correct, trials)
        labels[task_id] = label
        detail[task_id] = n_correct

        if progress_fn:
            progress_fn(i + 1, len(tasks), label)

    return {
        "model": model,
        "trials": trials,
        "temperature": temperature,
        "created": time.time(),
        "labels": labels,
        "n_correct": detail,
    }


def split_tasks(tasks, pilot_fraction, seed):
    """Deterministic pilot/eval split."""
    import random as _random
    shuffled = list(tasks)
    _random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * pilot_fraction))
    return {"pilot": shuffled[:cut], "eval": shuffled[cut:]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _progress(done, total, label):
    mark = {"required": "R", "not_required": ".", "ambiguous": "?"}.get(label, "-")
    end = "\n" if done == total else ""
    print(mark, end=end, flush=True)
    if done % 50 == 0 and done != total:
        print(f" {done}/{total}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Label tool_necessity using a local Ollama model.")
    p.add_argument("--dataset", default="gsm8k",
                   help="gsm8k | humaneval | coqa | overruling | headlines")
    p.add_argument("--limit", type=int, default=50,
                   help="tasks to load before splitting (default 50)")
    p.add_argument("--trials", type=int, default=3,
                   help="attempts per task (default 3)")
    p.add_argument("--pilot-fraction", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--model", default="llama3.2:3b",
                   help="Ollama model name (default llama3.2:3b)")
    p.add_argument("--ollama-url", default="http://localhost:11434/v1",
                   help="Ollama API base URL")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="output path (default labels/<dataset>.json)")
    args = p.parse_args(argv)

    # Check Ollama is running
    client = _make_client(args.ollama_url)
    try:
        client.models.list()
    except Exception as exc:
        print(f"\nCan't reach Ollama at {args.ollama_url}.\n"
              f"Make sure it's running: ollama serve\n"
              f"Error: {exc}", file=sys.stderr)
        return 1

    from task_datasets import load_dataset
    tasks = load_dataset(args.dataset, max_samples=args.limit)
    if not tasks:
        print(f"No tasks loaded for {args.dataset!r}.", file=sys.stderr)
        return 1

    split = split_tasks(tasks, args.pilot_fraction, args.seed)
    pilot = split["pilot"]

    print(f"{args.dataset}: {len(tasks)} loaded -> {len(pilot)} pilot / "
          f"{len(split['eval'])} held out | {args.model} | {args.trials} trials")
    print("  R = needed a tool, . = did not, ? = inconsistent\n")

    label_set = label_tasks(
        client, pilot, args.model,
        trials=args.trials, temperature=args.temperature,
        max_tokens=args.max_tokens, progress_fn=_progress)

    out = args.out or f"labels/{args.dataset}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(label_set, fh, indent=2, sort_keys=True)

    counts = {t.value: 0 for t in ToolNecessity}
    for label in label_set["labels"].values():
        counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values()) or 1

    print(f"\n  required      {counts['required']:>4}  ({counts['required']/total*100:.0f}%)")
    print(f"  not_required  {counts['not_required']:>4}  ({counts['not_required']/total*100:.0f}%)")
    print(f"  ambiguous     {counts['ambiguous']:>4}  ({counts['ambiguous']/total*100:.0f}%)")
    print(f"\n  -> {out}")

    gradeable = counts["required"] + counts["not_required"]
    if gradeable and counts["required"] / gradeable < 0.10:
        print(f"\n  WARNING: only {counts['required']} of {gradeable} gradeable tasks "
              f"needed a tool. This dataset may be ceilinged for {args.model} — "
              f"consider a harder dataset or a smaller model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
