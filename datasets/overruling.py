"""
overruling.py
=============

Loader for the Overruling dataset (LegalBench / CaseHOLD).

Source: nguha/legalbench on HuggingFace (overruling config, test split)
        Originally from Stanford RegLab's CaseHOLD benchmark.

~2,400 sentences from US caselaw, binary-labeled: does this sentence
overrule a previous case? (Yes / No)

Task: binary classification requiring legal reasoning.
Tool type: retrieval — most LLMs can pattern-match "overruling language"
    from parametric knowledge, but tricky cases benefit from retrieving
    the referenced precedent.

Small dataset (2,400 sentences), so we fetch it in one paginated pass
through the HuggingFace rows API.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from typing import List

from .base import DatasetLoader, TaskItem, register

_HF_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=nguha/legalbench&config=overruling&split=test"
)

_PAGE_SIZE = 100
_TOTAL_ROWS = 2394


@register("overruling")
class OverrulingLoader(DatasetLoader):

    @property
    def tool_type(self) -> str:
        return "retrieval"

    @property
    def description(self) -> str:
        return "LegalBench overruling detection (~2.4k sentences, binary classification)"

    def download(self) -> None:
        cache = self.data_dir / "overruling.jsonl"
        if cache.exists():
            return

        print(f"  Downloading overruling via HuggingFace rows API ...", file=sys.stderr)
        rows_collected = 0
        tmp = cache.with_suffix(".tmp")

        with open(tmp, "w", encoding="utf-8") as fh:
            offset = 0
            while offset < _TOTAL_ROWS:
                length = min(_PAGE_SIZE, _TOTAL_ROWS - offset)
                url = f"{_HF_ROWS_URL}&offset={offset}&length={length}"
                req = urllib.request.Request(url, headers={"User-Agent": "confidence-tool-routing/0.1"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                rows = data.get("rows", [])
                if not rows:
                    break

                for entry in rows:
                    row = entry.get("row", {})
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_collected += 1

                offset += len(rows)

        tmp.rename(cache)
        print(f"  Downloaded {rows_collected} overruling sentences to {cache}", file=sys.stderr)

    def parse(self) -> List[TaskItem]:
        cache = self.data_dir / "overruling.jsonl"
        rows = self._read_jsonl(cache)

        items: List[TaskItem] = []
        for i, row in enumerate(rows):
            text = row.get("text", "").strip()
            label = row.get("answer", "").strip()
            if not text or not label:
                continue
            items.append(TaskItem(
                task_id=f"overruling_{i:04d}",
                query=(
                    "Does the following sentence from a judicial opinion "
                    "overrule a previous case? Answer Yes or No.\n\n"
                    f"{text}"
                ),
                gold_answer=label,
                dataset="overruling",
                tool_type="retrieval",
                meta={
                    "sentence": text,
                    "label": label,
                },
            ))
        return items
