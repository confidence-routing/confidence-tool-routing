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
from pathlib import Path
from typing import Iterator, List, Optional

from .models import TaskRecord


class RunLogger:
    def __init__(self, run_id: str, log_dir: str = "runs"):
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"{run_id}.jsonl"
        # open in append mode so resuming a run doesn't clobber prior rows
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
