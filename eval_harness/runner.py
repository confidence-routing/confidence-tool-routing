"""
runner.py
=========

The loop that ties everything together:

    question -> answer -> confidence -> route -> tool -> grade -> log

Every other module is a piece of this and none of them import each
other. runner.py is the only place that knows the order.

One TaskRecord comes out per task, with the cost of each stage recorded
separately -- the answer, the confidence estimate, and the tool
round-trip -- because the point of the experiment is that those three
differ, and a single lumped total cannot show it.

Token counts come from what the provider reported in ``usage``, never
from re-tokenizing locally: a provider that counts differently than
tiktoken would otherwise skew CPST in a way nothing in the logs explains.

The client is duck-typed on purpose. Anything with ``.complete()`` and
``.sample()`` works, which is how the tests drive the whole loop with a
scripted stub and no key.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .confidence import (
    ConfidenceMethod,
    _get,
    build_verifier_messages,
    estimate_entropy_confidence,
    estimate_external_verifier_confidence,
    estimate_hybrid_confidence,
    estimate_self_consistency_confidence,
)
from .costs import PRICING_TABLE, estimate_cost_usd
from .grading import grade
from .models import (
    LatencyBreakdown,
    RoutingDecision,
    TaskRecord,
    ToolNecessity,
    ToolType,
)
from .router import route
from .tools import calculate, run_python

# Which tool each dataset's TOOL branch reaches for.
DATASET_TOOL = {"gsm8k": ToolType.CALCULATOR, "humaneval": ToolType.CODE_EXECUTOR}

_ANSWER_SYSTEM = (
    "Answer the question. Be brief. State your final answer on the last "
    "line, with no explanation after it."
)
_EXPRESSION_SYSTEM = (
    "Rewrite the arithmetic needed to answer this question as ONE Python "
    "expression using only numbers and + - * / ( ). Output the expression "
    "alone, nothing else."
)


@dataclass
class RunConfig:
    model: str = "llama-3.3-70b"
    verifier_model: str = "llama-3.1-8b"
    method: ConfidenceMethod = ConfidenceMethod.ENTROPY
    threshold: float = 0.7
    k: int = 5                       # self-consistency samples
    sample_temperature: float = 0.8
    max_tokens: int = 512
    seed: Optional[int] = 0
    on_missing: str = "tool"
    tool_timeout: float = 10.0
    run_id: str = "run"

    def uses_self_consistency(self) -> bool:
        return self.method in (ConfidenceMethod.SELF_CONSISTENCY, ConfidenceMethod.HYBRID)

    def uses_verifier(self) -> bool:
        return self.method in (ConfidenceMethod.EXTERNAL_LLM, ConfidenceMethod.HYBRID)


def _text(response: Any) -> str:
    """The assistant's message text off a chat completion."""
    choices = _get(response, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        return ""
    return (_get(_get(choices[0], "message"), "content") or "").strip()


def _usage(response: Any) -> Dict[str, int]:
    usage = _get(response, "usage")
    return {
        "prompt": int(_get(usage, "prompt_tokens") or 0),
        "completion": int(_get(usage, "completion_tokens") or 0),
    }


@dataclass
class _Cost:
    """Running token totals for one task, kept per stage."""
    prompt: int = 0
    completion: int = 0

    def add(self, response: Any) -> None:
        u = _usage(response)
        self.prompt += u["prompt"]
        self.completion += u["completion"]


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def estimate_confidence(
    client: Any, task: Dict[str, Any], answer_response: Any, answer_text: str,
    cfg: RunConfig,
) -> Dict[str, Any]:
    """
    Run whichever estimators the config asks for.

    Returns the score plus the tokens each estimator spent, split by
    which model spent them: self-consistency resamples the main model,
    the verifier is a deliberately cheaper one, and pricing both at one
    rate would misreport whichever is not that model.
    """
    messages = [{"role": "system", "content": _ANSWER_SYSTEM},
                {"role": "user", "content": task["query"]}]

    signals: Dict[Any, Any] = {}
    main_cost, verifier_cost = _Cost(), _Cost()

    # (a) token entropy -- free, the logprobs are already on the answer.
    entropy = estimate_entropy_confidence(answer_response)
    signals[ConfidenceMethod.ENTROPY] = entropy

    # (b) self-consistency -- k separate calls, main model.
    if cfg.uses_self_consistency():
        samples = client.sample(messages, cfg.model, k=cfg.k,
                                temperature=cfg.sample_temperature,
                                max_tokens=cfg.max_tokens)
        for s in samples:
            main_cost.add(s)
        signals[ConfidenceMethod.SELF_CONSISTENCY] = \
            estimate_self_consistency_confidence(samples)

    # (c) external verifier -- one call, cheap model, one-token verdict.
    if cfg.uses_verifier():
        verdict = client.complete(
            build_verifier_messages(task["query"], answer_text),
            cfg.verifier_model, max_tokens=1, temperature=0.0,
            logprobs=True, top_logprobs=5,
        )
        verifier_cost.add(verdict)
        signals[ConfidenceMethod.EXTERNAL_LLM] = \
            estimate_external_verifier_confidence(verdict)

    if cfg.method == ConfidenceMethod.HYBRID:
        score = getattr(estimate_hybrid_confidence(signals), "confidence", None)
    else:
        score = getattr(signals.get(cfg.method), "confidence", None)

    return {"score": score, "main": main_cost, "verifier": verifier_cost,
            "signals": {str(getattr(k, "value", k)): getattr(v, "confidence", None)
                        for k, v in signals.items()}}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def apply_tool(
    client: Any, task: Dict[str, Any], answer_text: str, cfg: RunConfig,
) -> Dict[str, Any]:
    """
    Take the TOOL branch. Returns the post-tool answer and its cost.

    The tool does not replace the model, it corrects it: the calculator
    evaluates arithmetic the model wrote, the runner tells the model
    which of its own tests failed. A tool that silently substituted its
    own answer would make the routing decision unfalsifiable -- every
    TOOL task would score the tool, not the routing.

    If the tool cannot help (an expression that will not parse, code that
    already passes), the direct answer stands. Falling back is not a
    failure: it is the honest outcome of an unnecessary tool call, and
    the cost of trying is still billed.
    """
    dataset = task.get("dataset", "")
    cost = _Cost()

    if dataset == "gsm8k":
        response = client.complete(
            [{"role": "system", "content": _EXPRESSION_SYSTEM},
             {"role": "user", "content": task["query"]}],
            cfg.model, max_tokens=64, temperature=0.0,
        )
        cost.add(response)
        computed = calculate(_text(response))
        return {"answer": computed if computed is not None else answer_text,
                "cost": cost, "tool_used": ToolType.CALCULATOR,
                "tool_helped": computed is not None}

    if dataset == "humaneval":
        from .grading import build_humaneval_program
        program = build_humaneval_program(answer_text, task.get("meta", {}))
        if program is None:
            return {"answer": answer_text, "cost": cost,
                    "tool_used": ToolType.CODE_EXECUTOR, "tool_helped": False}

        result = run_python(program, timeout=cfg.tool_timeout)
        if result.ok:
            # Already correct; execution confirmed it and changed nothing.
            return {"answer": answer_text, "cost": cost,
                    "tool_used": ToolType.CODE_EXECUTOR, "tool_helped": False}

        error = (result.stderr or "")[-600:]
        retry = client.complete(
            [{"role": "system", "content": _ANSWER_SYSTEM},
             {"role": "user", "content": task["query"]},
             {"role": "assistant", "content": answer_text},
             {"role": "user", "content":
              f"Running your code against the tests failed:\n\n{error}\n\n"
              f"Return the corrected function only."}],
            cfg.model, max_tokens=cfg.max_tokens, temperature=0.0,
        )
        cost.add(retry)
        return {"answer": _text(retry) or answer_text, "cost": cost,
                "tool_used": ToolType.CODE_EXECUTOR, "tool_helped": True}

    raise KeyError(f"No tool wired for dataset {dataset!r}. Add one in runner.py.")


# ---------------------------------------------------------------------------
# One task
# ---------------------------------------------------------------------------

def run_task(client: Any, task: Dict[str, Any], cfg: RunConfig) -> TaskRecord:
    """Answer, estimate, route, maybe use a tool, grade, and return the row."""
    dataset = task.get("dataset", "")
    messages = [{"role": "system", "content": _ANSWER_SYSTEM},
                {"role": "user", "content": task["query"]}]

    t0 = time.perf_counter()
    answer_response = client.complete(
        messages, cfg.model, max_tokens=cfg.max_tokens,
        temperature=0.0, seed=cfg.seed, logprobs=True, top_logprobs=5,
    )
    answer_text = _text(answer_response)
    answer_usage = _usage(answer_response)
    t_answer = time.perf_counter()

    conf = estimate_confidence(client, task, answer_response, answer_text, cfg)
    t_conf = time.perf_counter()

    decision = route(conf["score"], cfg.threshold, on_missing=cfg.on_missing)

    final_answer, tool_cost = answer_text, _Cost()
    tool_used, tool_helped = ToolType.NONE, False
    if decision == RoutingDecision.TOOL:
        outcome = apply_tool(client, task, answer_text, cfg)
        final_answer = outcome["answer"]
        tool_cost = outcome["cost"]
        tool_used = outcome["tool_used"]
        tool_helped = outcome["tool_helped"]
    t_tool = time.perf_counter()

    correct = grade(dataset, final_answer,
                    gold_answer=task.get("gold_answer"),
                    meta=task.get("meta"))

    # The confidence stage can span two models: self-consistency resamples
    # the main one, the verifier is a cheaper one. The token fields carry
    # the main-model side; the verifier is priced into the flat field, which
    # is what it exists for. Its raw tokens go to meta either way, so the
    # split stays auditable even when the model is missing from the table.
    verifier_usd = 0.0
    if conf["verifier"].prompt or conf["verifier"].completion:
        if cfg.verifier_model in PRICING_TABLE:
            verifier_usd = estimate_cost_usd(
                cfg.verifier_model, conf["verifier"].prompt, conf["verifier"].completion)
        else:
            warnings.warn(
                f"No pricing entry for verifier model {cfg.verifier_model!r}; its "
                f"tokens are logged in meta but excluded from CPST. Add it to "
                f"PRICING_TABLE in costs.py.", UserWarning, stacklevel=2)

    return TaskRecord(
        task_id=str(task.get("task_id", "")),
        dataset=dataset,
        run_id=cfg.run_id,
        seed=cfg.seed or 0,
        query=task.get("query", ""),
        gold_answer=task.get("gold_answer"),
        model_answer=final_answer,
        correct=correct,
        tool_necessity=ToolNecessity(task.get("tool_necessity", "not_required")),
        routing_decision=decision,
        tool_called=decision == RoutingDecision.TOOL,
        tool_used=tool_used,
        confidence_score=conf["score"],
        confidence_method=cfg.method,
        confidence_threshold=cfg.threshold,
        model_name=cfg.model,
        prompt_tokens=answer_usage["prompt"],
        completion_tokens=answer_usage["completion"],
        tool_prompt_tokens=tool_cost.prompt,
        tool_completion_tokens=tool_cost.completion,
        confidence_prompt_tokens=conf["main"].prompt,
        confidence_completion_tokens=conf["main"].completion,
        confidence_model_name=cfg.model if conf["main"].prompt else "",
        confidence_api_cost_usd=verifier_usd,
        latency=LatencyBreakdown(
            main_model_ms=(t_answer - t0) * 1000,
            confidence_ms=(t_conf - t_answer) * 1000,
            tool_ms=(t_tool - t_conf) * 1000,
        ),
        meta={
            "signals": conf["signals"],
            "tool_helped": tool_helped,
            "verifier_model": cfg.verifier_model if conf["verifier"].prompt else None,
            "verifier_prompt_tokens": conf["verifier"].prompt,
            "verifier_completion_tokens": conf["verifier"].completion,
            "task_tool_category": task.get("tool_type", "none"),
        },
    )


def run_dataset(
    client: Any, tasks: Sequence[Dict[str, Any]], cfg: RunConfig,
    logger: Any = None, on_error: str = "skip",
    progress: Optional[Callable[[int, int, Optional[TaskRecord]], None]] = None,
) -> List[TaskRecord]:
    """
    Run every task, logging each row as it completes.

    Rows are written one at a time rather than at the end, so a sweep
    that dies two thirds through leaves two thirds of a usable log
    instead of nothing -- the reason RunLogger is append-only JSONL.

    on_error="skip" keeps going past a task that raises, which is the
    right default for a long paid run: one malformed record should not
    cost you the other 299. Failures are counted and reported rather
    than swallowed.
    """
    if on_error not in ("skip", "raise"):
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")

    records: List[TaskRecord] = []
    failures: List[str] = []

    for i, task in enumerate(tasks):
        try:
            record = run_task(client, task, cfg)
        except Exception as exc:
            if on_error == "raise":
                raise
            failures.append(f"{task.get('task_id', i)}: {exc}")
            if progress:
                progress(i + 1, len(tasks), None)
            continue

        records.append(record)
        if logger is not None:
            logger.log(record)
        if progress:
            progress(i + 1, len(tasks), record)

    if failures:
        warnings.warn(
            f"{len(failures)} of {len(tasks)} tasks failed and were skipped. "
            f"First: {failures[0]}", UserWarning, stacklevel=2)
    return records
