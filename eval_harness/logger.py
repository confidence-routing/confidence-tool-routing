"""
logger.py
=========

Minimal append-only JSONL logger for TaskRecords.

One line per task, one file per run. JSONL (not a single JSON array) so a
crashed run is still readable up to the last completed line, and so runs
can be tailed / streamed / concatenated with plain shell tools.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Iterator, List, Optional

from .models import TaskRecord


class RunLogger:
    def __init__(self, run_id: str, log_dir: str = "runs", *, resume: bool = False):
        """
        Open (or extend) the log for ``run_id``.

        Append mode is deliberate: a crashed run keeps every row it
        managed to write, and can be continued. The cost is that
        re-running a config under a run_id that already has rows appends
        to them instead of replacing them, and every metric is then
        computed over both runs at once -- a doubled task count, a
        success rate averaged across two configurations, and nothing
        anywhere saying so. The numbers stay plausible, which is what
        makes it dangerous.

        So an existing non-empty log warns unless ``resume=True`` says
        the append was intended. Nothing is deleted either way: pick a
        fresh run_id, or remove the file yourself if you meant to
        replace it.
        """
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"{run_id}.jsonl"

        if not resume and self.path.exists() and self.path.stat().st_size > 0:
            warnings.warn(
                f"{self.path} already has rows and this run will append to "
                f"them, so metrics would cover both runs at once. Pass "
                f"resume=True if that is intended, or use a different "
                f"run_id.",
                UserWarning,
                stacklevel=2,
            )

        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, record: TaskRecord) -> None:
        record.run_id = record.run_id or self.run_id
        line = json.dumps(record.to_dict(), ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @staticmethod
    def load(path: str) -> List[TaskRecord]:
        """Read every record back from a JSONL log file."""
        records: List[TaskRecord] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(TaskRecord.from_dict(json.loads(line)))
        return records

    @staticmethod
    def iter_load(path: str) -> Iterator[TaskRecord]:
        """Same as load(), but streams records instead of materializing a list.
        Useful once a run has thousands of tasks."""
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield TaskRecord.from_dict(json.loads(line))
