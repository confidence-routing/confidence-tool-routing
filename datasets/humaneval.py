"""
humaneval.py
============

Loader for HumanEval — OpenAI code generation benchmark.

Source: openai/human-eval on GitHub (gzipped JSONL)
        https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz

164 hand-written Python programming tasks. Each has a function signature
+ docstring (the prompt) and a test suite for verification.

Task: generate the function body that passes all tests.
Tool type: code_executor — the model writes code; running it in a
    sandbox verifies correctness. Tool necessity is clear: can the model
    produce correct code without test-execution feedback?
"""

from __future__ import annotations

import gzip
import json
import sys
from typing import List

from .base import DatasetLoader, TaskItem, register

_HUMANEVAL_URL = (
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
)


@register("humaneval")
class HumanEvalLoader(DatasetLoader):

    @property
    def tool_type(self) -> str:
        return "code_executor"

    @property
    def description(self) -> str:
        return "HumanEval code generation (164 Python tasks)"

    def download(self) -> None:
        gz_path = self.data_dir / "HumanEval.jsonl.gz"
        jsonl_path = self.data_dir / "HumanEval.jsonl"

        if jsonl_path.exists():
            return

        self._download_file(_HUMANEVAL_URL, "HumanEval.jsonl.gz")

        # Decompress
        print(f"  Decompressing {gz_path} ...", file=sys.stderr)
        with gzip.open(gz_path, "rt", encoding="utf-8") as gz, \
             open(jsonl_path, "w", encoding="utf-8") as out:
            out.write(gz.read())

    def parse(self) -> List[TaskItem]:
        rows = self._read_jsonl(self.data_dir / "HumanEval.jsonl")

        items: List[TaskItem] = []
        for row in rows:
            task_id = row.get("task_id", "")
            prompt = row.get("prompt", "").strip()
            canonical_solution = row.get("canonical_solution", "")
            test_code = row.get("test", "")
            entry_point = row.get("entry_point", "")

            if not prompt or not task_id:
                continue

            items.append(TaskItem(
                task_id=task_id.replace("/", "_"),
                query=f"Complete the following Python function:\n\n{prompt}",
                gold_answer=canonical_solution.strip(),
                dataset="humaneval",
                tool_type="code_executor",
                meta={
                    "original_task_id": task_id,
                    "prompt": prompt,
                    "canonical_solution": canonical_solution,
                    "test_code": test_code,
                    "entry_point": entry_point,
                },
            ))
        return items
