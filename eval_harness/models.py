"""
models.py
=========

Defines the single unit of logging for the eval harness: TaskRecord.

Every task the system processes — whether it answered directly or called a
tool — produces exactly one TaskRecord. Everything in metrics.py and
report.py is computed by aggregating a list of these.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any
import time


class ToolType(str, Enum):
    NONE = "none"
    RETRIEVAL = "retrieval"
    CALCULATOR = "calculator"
    WEB_SEARCH = "web_search"
    CODE_EXECUTOR = "code_executor"


class ConfidenceMethod(str, Enum):
    ENTROPY = "entropy"
    SELF_CONSISTENCY = "self_consistency"
    EXTERNAL_LLM = "external_llm"
    HYBRID = "hybrid"
    FIXED = "fixed"  # used by the "no confidence estimator" ablation


class ToolNecessity(str, Enum):
    """Ground-truth label: did this task actually need a tool to answer
    correctly? Set by the dataset / task designer, not by the system."""
    REQUIRED = "required"
    NOT_REQUIRED = "not_required"
    AMBIGUOUS = "ambiguous"  # excluded from precision/recall by default


class RoutingDecision(str, Enum):
    DIRECT = "direct"
    TOOL = "tool"


@dataclass
class LatencyBreakdown:
    main_model_ms: float = 0.0
    confidence_ms: float = 0.0
    tool_ms: float = 0.0
    total_ms: float = 0.0

    def __post_init__(self):
        if self.total_ms == 0.0:
            self.total_ms = self.main_model_ms + self.confidence_ms + self.tool_ms


@dataclass
class TaskRecord:
    # --- identity ---
    task_id: str
    dataset: str                     # e.g. "gsm8k", "natural_questions", "humaneval"
    run_id: str = ""
    seed: int = 0
    timestamp: float = field(default_factory=time.time)

    # --- inputs / outputs ---
    query: str = ""
    gold_answer: Optional[str] = None
    model_answer: Optional[str] = None
    correct: Optional[bool] = None   # filled in by task-specific grading

    # --- ground truth for routing evaluation ---
    tool_necessity: ToolNecessity = ToolNecessity.NOT_REQUIRED

    # --- routing decision actually taken ---
    routing_decision: RoutingDecision = RoutingDecision.DIRECT
    tool_called: bool = False
    tool_used: ToolType = ToolType.NONE

    # --- confidence signal ---
    confidence_score: Optional[float] = None      # in [0, 1]
    confidence_method: ConfidenceMethod = ConfidenceMethod.FIXED
    confidence_threshold: float = 0.7

    # --- cost accounting ---
    model_name: str = "gpt-4o-mini"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_prompt_tokens: int = 0       # extra tokens spent because of tool round-trip
    tool_completion_tokens: int = 0
    tool_api_cost_usd: float = 0.0    # non-token costs, e.g. a paid search API call

    # --- latency ---
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)

    # --- free-form extras (ablation tag, experiment name, etc.) ---
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_prompt_tokens(self) -> int:
        return self.prompt_tokens + self.tool_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self.completion_tokens + self.tool_completion_tokens

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # enums -> plain strings for JSON
        d["tool_necessity"] = self.tool_necessity.value
        d["routing_decision"] = self.routing_decision.value
        d["tool_used"] = self.tool_used.value
        d["confidence_method"] = self.confidence_method.value
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TaskRecord":
        d = dict(d)
        d["tool_necessity"] = ToolNecessity(d.get("tool_necessity", "not_required"))
        d["routing_decision"] = RoutingDecision(d.get("routing_decision", "direct"))
        d["tool_used"] = ToolType(d.get("tool_used", "none"))
        d["confidence_method"] = ConfidenceMethod(d.get("confidence_method", "fixed"))
        latency = d.get("latency") or {}
        d["latency"] = LatencyBreakdown(**latency) if isinstance(latency, dict) else latency
        return TaskRecord(**d)
