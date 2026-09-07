"""
datasets
========

Dataset loaders for the confidence-based tool routing project.

Each loader downloads its dataset on first use (into ``data/<name>/``),
parses it into a uniform list of ``TaskItem`` dicts, and caches the
parsed result so subsequent calls are instant.

Every TaskItem has at minimum:
    task_id     str   -- unique identifier
    query       str   -- the question / input the model will see
    gold_answer str   -- the correct answer for grading
    dataset     str   -- which dataset this came from
    tool_type   str   -- the tool category this task exercises
    meta        dict  -- dataset-specific extras

Loaders:
    headlines.py    News Category Dataset (Kaggle/HuffPost) — classification
    coqa.py         CoQA (Stanford) — conversational QA / retrieval
    overruling.py   Overruling (LegalBench/CaseHOLD) — legal binary classification
    gsm8k.py        GSM8K (OpenAI) — math word problems / calculator
    humaneval.py    HumanEval (OpenAI) — code generation / code executor
"""

from .base import TaskItem, DatasetLoader, DATASET_REGISTRY, load_dataset, list_datasets

__all__ = [
    "TaskItem",
    "DatasetLoader",
    "DATASET_REGISTRY",
    "load_dataset",
    "list_datasets",
]
