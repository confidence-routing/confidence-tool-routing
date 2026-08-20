"""
eval_harness
============

Custom evaluation harness for the confidence-based tool routing project.

Core pieces:
    - models.py   TaskRecord: the single unit of logging (one task = one row)
    - costs.py    OpenAI pricing table + token/cost helpers
    - logger.py   Append-only JSONL run logger
    - metrics.py  CPST, ECE (per-tool + aggregate), routing precision/recall,
                  unnecessary/missed tool-call rate, latency breakdown
    - report.py   Turns a run's records into a summary dict / printable report
"""

from .models import TaskRecord, ToolType, ToolNecessity
from .costs import estimate_cost_usd
from .logger import RunLogger
from .metrics import (
    compute_cpst,
    compute_ece,
    compute_routing_precision_recall,
    compute_unnecessary_call_rate,
    compute_missed_call_rate,
    compute_latency_breakdown,
)
from .report import build_report, print_report

__all__ = [
    "TaskRecord",
    "ToolType",
    "ToolNecessity",
    "estimate_cost_usd",
    "RunLogger",
    "compute_cpst",
    "compute_ece",
    "compute_routing_precision_recall",
    "compute_unnecessary_call_rate",
    "compute_missed_call_rate",
    "compute_latency_breakdown",
    "build_report",
    "print_report",
]
