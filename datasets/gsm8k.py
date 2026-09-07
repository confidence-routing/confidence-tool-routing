"""
gsm8k.py
========

Loader for GSM8K — Grade School Math 8K.

Source: OpenAI's grade-school-math repo on GitHub (raw JSONL)
        https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl

1,319 test questions (we use the test split since the training set would
only be needed to fine-tune, which we don't do). Each is a multi-step
arithmetic word problem with a chain-of-thought solution ending in
``#### <number>``.

Task: solve the word problem and produce the final numeric answer.
Tool type: calculator — the model either does the arithmetic correctly
    in its head or it doesn't. Clean tool-necessity split: some are
    trivial mental math, others need multi-step calculation.
"""

from __future__ import annotations

import re
from typing import List

from .base import DatasetLoader, TaskItem, register

_GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math"
    "/master/grade_school_math/data/test.jsonl"
)


def _extract_final_answer(answer_text: str) -> str:
    """Pull the number after ``####`` from a GSM8K answer string."""
    match = re.search(r"####\s*(.+)$", answer_text, re.MULTILINE)
    if match:
        return match.group(1).strip().replace(",", "")
    return answer_text.strip()


@register("gsm8k")
class GSM8KLoader(DatasetLoader):

    @property
    def tool_type(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "GSM8K math word problems (1,319 test, multi-step arithmetic)"

    def download(self) -> None:
        self._download_file(_GSM8K_TEST_URL, "test.jsonl")

    def parse(self) -> List[TaskItem]:
        rows = self._read_jsonl(self.data_dir / "test.jsonl")

        items: List[TaskItem] = []
        for i, row in enumerate(rows):
            question = row.get("question", "").strip()
            answer_raw = row.get("answer", "")
            if not question:
                continue

            final_answer = _extract_final_answer(answer_raw)

            items.append(TaskItem(
                task_id=f"gsm8k_{i:04d}",
                query=question,
                gold_answer=final_answer,
                dataset="gsm8k",
                tool_type="calculator",
                meta={
                    "chain_of_thought": answer_raw,
                    "final_answer_numeric": final_answer,
                },
            ))
        return items
