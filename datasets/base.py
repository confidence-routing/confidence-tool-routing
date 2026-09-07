"""
base.py
=======

Base interface and registry for dataset loaders.

Every loader subclasses ``DatasetLoader`` and registers itself via
``@register``. The ``load_dataset("name")`` entry point handles
discovery, download, and parsing — callers never import individual
loader modules directly.

TaskItem is a plain TypedDict so it serializes to JSON without ceremony
and stays compatible with the eval harness's TaskRecord (one TaskItem
becomes one TaskRecord once the ground-truth pipeline fills in
confidence scores and routing decisions).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type
from typing import TypedDict


class TaskItem(TypedDict, total=False):
    """One task ready for the ground-truth labeling pipeline."""
    task_id: str
    query: str
    gold_answer: str
    dataset: str
    tool_type: str          # none | retrieval | calculator | code_executor | web_search
    meta: Dict[str, Any]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Type["DatasetLoader"]] = {}


def register(name: str):
    """Class decorator that adds a loader to the registry."""
    def decorator(cls: Type[DatasetLoader]):
        DATASET_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


class DatasetLoader(ABC):
    """Download + parse one dataset into a list of TaskItems."""

    name: str = ""

    def __init__(self, data_dir: Optional[Path] = None, max_samples: Optional[int] = None):
        self.data_dir = (data_dir or DATA_ROOT / self.name)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples

    # -- subclass contract --------------------------------------------------

    @abstractmethod
    def download(self) -> None:
        """Download raw files into self.data_dir. Skip if already present."""

    @abstractmethod
    def parse(self) -> List[TaskItem]:
        """Parse raw files into TaskItems."""

    @property
    @abstractmethod
    def tool_type(self) -> str:
        """Primary tool category for this dataset."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line human description."""

    # -- public API ---------------------------------------------------------

    def load(self, shuffle_seed: int = 42) -> List[TaskItem]:
        """Download (if needed), parse, shuffle deterministically, optionally truncate.

        Shuffling ensures that a ``max_samples`` slice is representative
        even when the raw data is sorted by label (e.g. Overruling has all
        Yes before all No).
        """
        import random as _random

        self.download()
        items = self.parse()

        rng = _random.Random(shuffle_seed)
        rng.shuffle(items)

        if self.max_samples is not None and len(items) > self.max_samples:
            items = items[:self.max_samples]
        return items

    # -- helpers for subclasses ---------------------------------------------

    def _download_file(self, url: str, filename: str, expected_sha256: Optional[str] = None) -> Path:
        """Download ``url`` to ``self.data_dir / filename`` if not present.

        Optionally verify a SHA-256 hash for reproducibility.
        """
        dest = self.data_dir / filename
        if dest.exists():
            if expected_sha256 and not self._check_sha256(dest, expected_sha256):
                print(f"  SHA-256 mismatch for {dest}, re-downloading...", file=sys.stderr)
                dest.unlink()
            else:
                return dest

        print(f"  Downloading {url} ...", file=sys.stderr)
        tmp = dest.with_suffix(".tmp")
        req = urllib.request.Request(url, headers={"User-Agent": "confidence-tool-routing/0.1"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                fh.write(chunk)
        tmp.rename(dest)

        if expected_sha256 and not self._check_sha256(dest, expected_sha256):
            raise RuntimeError(
                f"SHA-256 mismatch after download: {dest}. "
                f"Expected {expected_sha256}."
            )
        return dest

    @staticmethod
    def _check_sha256(path: Path, expected: str) -> bool:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest() == expected

    @staticmethod
    def _read_json(path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _ensure_loaders_imported() -> None:
    """Import all loader modules so they register themselves."""
    from . import headlines, coqa, overruling, gsm8k, humaneval  # noqa: F401


def load_dataset(
    name: str,
    max_samples: Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> List[TaskItem]:
    """Load a dataset by name. Downloads on first call."""
    _ensure_loaders_imported()
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    loader_cls = DATASET_REGISTRY[name]
    loader = loader_cls(data_dir=data_dir, max_samples=max_samples)
    return loader.load()


def list_datasets() -> List[Dict[str, str]]:
    """Return metadata about all registered datasets."""
    _ensure_loaders_imported()
    result = []
    for name, cls in sorted(DATASET_REGISTRY.items()):
        instance = cls()
        result.append({
            "name": name,
            "tool_type": instance.tool_type,
            "description": instance.description,
        })
    return result
