"""
ollama_quicktest.py
===================

Sanity-check: load a dataset, send a handful of tasks to a local Ollama
model, show the answers and grades side by side. Run this BEFORE
committing to a long labeling or experiment run.

    ollama pull llama3.2:3b

    python -m examples.ollama_quicktest --dataset gsm8k
    python -m examples.ollama_quicktest --dataset coqa --n 5 --model llama3.1:8b
    python -m examples.ollama_quicktest --dataset overruling --n 10
    python -m examples.ollama_quicktest --dataset humaneval --n 3

Shows for each task:
    - the query (truncated)
    - gold answer
    - model answer
    - correct / wrong
    - for CoQA: the token-F1 score

Also prints a summary: X/N correct, so you can tell at a glance if the
model is reasonable on this dataset before spending time on a full run.

Requires: ollama running locally, pip install openai
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness.grading import grade, token_f1


# ---------------------------------------------------------------------------
# Ollama client (same minimal approach as the label pilot)
# ---------------------------------------------------------------------------

def _make_client(base_url: str = "http://localhost:11434/v1"):
    try:
        from openai import OpenAI
    except ImportError:
        print("The 'openai' package is required. Install with:\n"
              "  pip install openai", file=sys.stderr)
        sys.exit(1)
    return OpenAI(base_url=base_url, api_key="ollama")


def _complete(client, model, messages, **kwargs):
    response = client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    return (response.choices[0].message.content or "").strip()


_ANSWER_SYSTEM = (
    "Answer the question. Be brief. State your final answer on the last "
    "line, with no explanation after it."
)

_COQA_SYSTEM = (
    "Answer the question in as few words as possible, using the wording of "
    "the source where you can. Do not explain."
)


def build_messages(task, *, include_passage=False):
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


def _trunc(text, width=80):
    text = " ".join((text or "").split())
    return text[:width] + "..." if len(text) > width else text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Quick sanity-check: send a few tasks to Ollama and show results.")
    p.add_argument("--dataset", default="gsm8k",
                   help="gsm8k | humaneval | coqa | overruling | headlines")
    p.add_argument("--n", type=int, default=5, help="number of tasks (default 5)")
    p.add_argument("--model", default="llama3.2:3b")
    p.add_argument("--ollama-url", default="http://localhost:11434/v1")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--with-passage", action="store_true",
                   help="for CoQA: include the passage (the 'with tool' condition)")
    args = p.parse_args(argv)

    client = _make_client(args.ollama_url)
    try:
        models = client.models.list()
        available = [m.id for m in models.data] if models.data else []
        print(f"Ollama models available: {', '.join(available) or '(none)'}")
    except Exception as exc:
        print(f"\nCan't reach Ollama at {args.ollama_url}.\n"
              f"Make sure it's running: ollama serve\n"
              f"Error: {exc}", file=sys.stderr)
        return 1

    from task_datasets import load_dataset
    tasks = load_dataset(args.dataset, max_samples=args.n)
    if not tasks:
        print(f"No tasks loaded for {args.dataset!r}.", file=sys.stderr)
        return 1

    print(f"\n{args.dataset}: {len(tasks)} tasks | model: {args.model}"
          f"{' | with passage' if args.with_passage else ''}\n")

    n_correct = 0
    bar = "-" * 70

    for i, task in enumerate(tasks):
        messages = build_messages(task, include_passage=args.with_passage)

        try:
            answer = _complete(client, args.model, messages,
                               temperature=0.0, max_tokens=args.max_tokens)
        except Exception as exc:
            print(f"[{i+1}] ERROR: {exc}")
            continue

        try:
            correct = grade(task["dataset"], answer,
                            gold_answer=task.get("gold_answer"),
                            meta=task.get("meta"))
        except KeyError as exc:
            print(f"[{i+1}] NO GRADER: {exc}")
            continue

        if correct:
            n_correct += 1

        mark = "✓" if correct else "✗"

        print(bar)
        print(f"[{i+1}/{len(tasks)}] {task['task_id']}  {mark}")
        print(f"  query: {_trunc(task['query'])}")
        print(f"  gold:  {_trunc(task.get('gold_answer', ''), 60)}")
        print(f"  model: {_trunc(answer, 60)}")

        if task["dataset"] == "coqa":
            f1 = token_f1(answer, task.get("gold_answer"))
            print(f"  F1:    {f1:.3f}")

    print(bar)
    print(f"\nSummary: {n_correct}/{len(tasks)} correct "
          f"({n_correct/len(tasks)*100:.0f}%)" if tasks else "")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
