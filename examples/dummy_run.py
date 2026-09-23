"""
dummy_run.py
============

Simulates a small batch of tasks across the four tool categories, logs
them through RunLogger, then reloads the log from disk and builds a full
report. This exists to prove the harness works end-to-end before any real
model calls are wired in — run it first, get a report, THEN plug in the
confidence module and tool router from Phase 2.

Run:
    python -m examples.dummy_run          (from the project root)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness import (
    TaskRecord, ToolType, ToolNecessity, RunLogger,
    build_report, print_report,
)
from eval_harness.models import ConfidenceMethod, RoutingDecision, LatencyBreakdown
from eval_harness.router import route, select_threshold, sweep_thresholds

RNG = random.Random(42)

DATASETS_BY_TOOL = {
    ToolType.RETRIEVAL: "natural_questions",
    ToolType.CALCULATOR: "gsm8k",
    ToolType.CODE_EXECUTOR: "humaneval",
    ToolType.WEB_SEARCH: "coqa",
}


def simulate_task(task_id: int, tool_category: ToolType) -> TaskRecord:
    """
    Fabricates one plausible task record. Confidence is deliberately
    correlated with correctness (with noise) so the ECE numbers in the
    report are meaningful rather than pure noise -- this is a stand-in
    for the real confidence module, not a substitute for it.
    """
    # ground truth: does this task actually need the tool?
    needs_tool = RNG.random() < 0.55

    # "true" latent quality of the model's direct answer
    true_quality = RNG.betavariate(3, 2) if not needs_tool else RNG.betavariate(1.5, 3)

    # noisy confidence estimate around the true quality (models are
    # typically a bit overconfident -- nudge the mean up slightly)
    confidence = min(1.0, max(0.0, true_quality + RNG.gauss(0.08, 0.12)))

    threshold = 0.7
    routed_to_tool = route(confidence, threshold) == RoutingDecision.TOOL

    tool_called = routed_to_tool
    tool_used = tool_category if tool_called else ToolType.NONE

    # correctness: direct answers succeed roughly at true_quality rate;
    # tool-assisted answers get a reliability boost when the tool was
    # actually needed, and are roughly as good as direct otherwise
    if tool_called:
        p_correct = min(0.97, true_quality + (0.25 if needs_tool else 0.05))
    else:
        p_correct = true_quality
    correct = RNG.random() < p_correct

    prompt_tokens = RNG.randint(120, 400)
    completion_tokens = RNG.randint(30, 180)
    tool_prompt_tokens = RNG.randint(150, 600) if tool_called else 0
    tool_completion_tokens = RNG.randint(40, 220) if tool_called else 0
    tool_api_cost = round(RNG.uniform(0.0005, 0.004), 5) if tool_used == ToolType.WEB_SEARCH else 0.0

    main_ms = RNG.uniform(300, 900)
    confidence_ms = RNG.uniform(20, 90)
    tool_ms = RNG.uniform(200, 1500) if tool_called else 0.0

    return TaskRecord(
        task_id=f"task_{task_id:04d}",
        dataset=DATASETS_BY_TOOL[tool_category],
        query=f"(simulated {tool_category.value} query #{task_id})",
        gold_answer="(gold)",
        model_answer="(prediction)",
        correct=correct,
        tool_necessity=ToolNecessity.REQUIRED if needs_tool else ToolNecessity.NOT_REQUIRED,
        routing_decision=RoutingDecision.TOOL if tool_called else RoutingDecision.DIRECT,
        tool_called=tool_called,
        tool_used=tool_used,
        confidence_score=round(confidence, 4),
        confidence_method=ConfidenceMethod.HYBRID,
        confidence_threshold=threshold,
        model_name="gpt-4o-mini",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tool_prompt_tokens=tool_prompt_tokens,
        tool_completion_tokens=tool_completion_tokens,
        tool_api_cost_usd=tool_api_cost,
        latency=LatencyBreakdown(main_model_ms=main_ms, confidence_ms=confidence_ms, tool_ms=tool_ms),
        meta={"task_tool_category": tool_category.value},
    )


def main() -> None:
    logger = RunLogger(run_id="dummy_run_001", log_dir="runs")

    tool_categories = list(DATASETS_BY_TOOL.keys())
    n_per_category = 60

    task_id = 0
    for category in tool_categories:
        for _ in range(n_per_category):
            record = simulate_task(task_id, category)
            logger.log(record)
            task_id += 1
    logger.close()

    print(f"Logged {task_id} tasks to {logger.path}\n")

    # reload from disk to prove the log round-trips cleanly, then report
    records = RunLogger.load(str(logger.path))
    report = build_report(
        records,
        ece_group_by=lambda r: r.meta.get("task_tool_category", r.tool_used.value),
    )
    print_report(report, title="Dummy Run — Confidence-Based Tool Routing")

    # What the router would have done at other cuts. The run above used
    # 0.7 throughout; this is how the operating point gets picked from a
    # pilot instead of by taste. No CPST column -- correctness under a
    # decision the run never took is not in the log. See router.py.
    print("\nThreshold sweep (routing quality only):")
    print(f"  {'thresh':>6}  {'prec':>5}  {'recall':>6}  {'f1':>5}  {'unnec':>6}  {'missed':>6}  {'tools':>5}")
    for pt in sweep_thresholds(records, [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]):
        fmt = lambda v: f"{v:.3f}" if v is not None else "  -  "
        print(f"  {pt.threshold:>6.2f}  {fmt(pt.precision):>5}  {fmt(pt.recall):>6}  "
              f"{fmt(pt.f1):>5}  {fmt(pt.unnecessary_call_rate):>6}  "
              f"{fmt(pt.missed_call_rate):>6}  {pt.n_tool:>5}")

    best = select_threshold(records)
    if best is not None:
        print(f"\n  best F1 at threshold {best.threshold:.2f} "
              f"(f1={best.f1:.3f}, {best.n_tool} tool calls of {len(records)})")


if __name__ == "__main__":
    main()
