"""
headlines.py
============

Loader for the News Category Dataset (HuffPost headlines).

Source: AiresPucrs/News-Category-Dataset on HuggingFace
        (mirror of Kaggle rmisra/news-category-dataset)

~210k headlines from HuffPost (2012–2022), each labeled with one of ~42
categories (U.S. NEWS, POLITICS, ENTERTAINMENT, SPORTS, etc.).

Task: classify the headline into its category.
Tool type: none / web_search — most headlines are classifiable from
    the text alone; the LLM's parametric knowledge suffices.

The HuggingFace datasets-server rows API returns JSON over HTTP and
requires no authentication or extra dependencies.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import List

from .base import DatasetLoader, TaskItem, register

_HF_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=AiresPucrs/News-Category-Dataset"
    "&config=default&split=train"
)

# The full dataset has ~209k rows. We paginate through the rows API.
_PAGE_SIZE = 100
_TOTAL_ROWS = 209_527  # from the API's num_rows_total


@register("headlines")
class HeadlinesLoader(DatasetLoader):

    @property
    def tool_type(self) -> str:
        return "none"

    @property
    def description(self) -> str:
        return "HuffPost headlines classification (~210k, 42 categories)"

    def download(self) -> None:
        cache = self.data_dir / "headlines.jsonl"
        if cache.exists():
            return

        print(f"  Downloading headlines via HuggingFace rows API ...", file=sys.stderr)
        total = self.max_samples if self.max_samples else _TOTAL_ROWS
        rows_collected = 0
        tmp = cache.with_suffix(".tmp")

        with open(tmp, "w", encoding="utf-8") as fh:
            offset = 0
            while offset < total:
                length = min(_PAGE_SIZE, total - offset)
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

                if rows_collected % 10_000 == 0:
                    print(f"    ... {rows_collected} rows", file=sys.stderr)

        tmp.rename(cache)
        print(f"  Downloaded {rows_collected} headlines to {cache}", file=sys.stderr)

    def parse(self) -> List[TaskItem]:
        cache = self.data_dir / "headlines.jsonl"
        rows = self._read_jsonl(cache)

        items: List[TaskItem] = []
        for i, row in enumerate(rows):
            text = row.get("text", "").strip()
            label = row.get("labels", "").strip()
            if not text or not label:
                continue
            items.append(TaskItem(
                task_id=f"headlines_{i:06d}",
                query=f"Classify the following news headline into its category:\n\n{text}",
                gold_answer=label,
                dataset="headlines",
                tool_type="none",
                meta={
                    "headline_text": text,
                    "category": label,
                },
            ))
        return items
