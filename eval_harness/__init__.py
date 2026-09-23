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
    - tools.py    What a TOOL decision calls: calculator, Python runner
    - grading.py  Fills in TaskRecord.correct, one grader per dataset
    - client.py   The only module that makes a network call
    - runner.py   The loop: answer, confidence, route, tool, grade, log
    - confidence.py  Confidence estimators: entropy, self-consistency,
                  external verifier, hybrid combiner
    - router.py   The routing decision itself + threshold selection
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
from .router import route, route_record, sweep_thresholds, select_threshold
from .tools import calculate, run_python
from .grading import grade, grade_gsm8k, grade_humaneval
# Safe at package level: client.py imports 'openai' lazily, inside
# Client.__post_init__, so the package stays importable without it.
from .client import Client, MissingAPIKey
from .runner import RunConfig, run_task, run_dataset

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
    "route",
    "route_record",
    "sweep_thresholds",
    "select_threshold",
    "calculate",
    "run_python",
    "grade",
    "grade_gsm8k",
    "grade_humaneval",
    "Client",
    "MissingAPIKey",
    "RunConfig",
    "run_task",
    "run_dataset",
]
