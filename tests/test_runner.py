"""
test_runner.py
==============

Drives the whole loop -- answer, confidence, route, tool, grade, log --
against a scripted stub client. No API key, no network, no cost.

That the loop is testable this way is the point of the client being
duck-typed: the stub is anything with .complete() and .sample(), so
these tests exercise the same runner.py code path a live run does.

Confidence is steered by handing the stub logprobs with a known shape:
one dominant candidate is near-certain (entropy ~0, confidence ~1), two
equally likely candidates is maximum uncertainty (normalized entropy 1,
confidence 0). That makes the routing outcome deterministic rather than
dependent on a real model's mood.

Run:
    python -m pytest tests/ -v
    (or, without pytest installed: python -m tests.test_runner)
"""

import math
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness.confidence import ConfidenceMethod
from eval_harness.models import RoutingDecision, ToolType
from eval_harness.runner import RunConfig, run_dataset, run_task

# log(1.0) ~ 0 -> one candidate carries all the mass -> confident
CONFIDENT = [{"token": "5", "logprob": -0.0001},
             {"token": "6", "logprob": -12.0}]
# two candidates at 0.5 each -> normalized entropy 1.0 -> confidence 0.0
UNSURE = [{"token": "5", "logprob": math.log(0.5)},
          {"token": "6", "logprob": math.log(0.5)}]


def response(text, top=None, prompt=100, completion=10):
    """An OpenAI-shaped chat completion, as a plain dict."""
    return {
        "choices": [{
            "message": {"content": text},
            "logprobs": {"content": [{"token": text[:1] or "x",
                                      "logprob": -0.05,
                                      "top_logprobs": top or CONFIDENT}]},
        }],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


class StubClient:
    """Returns scripted replies and records every call it received."""

    def __init__(self, answer="42", top=None, expression="6*7", retry=None):
        self.answer, self.top = answer, top
        self.expression, self.retry = expression, retry
        self.calls = []

    def complete(self, messages, model, **kw):
        system = messages[0].get("content", "")
        self.calls.append({"model": model, "system": system[:30], "kw": kw})

        if "ONE Python expression" in system:
            return response(self.expression, prompt=40, completion=8)
        if any("tests failed" in m.get("content", "") for m in messages):
            return response(self.retry or self.answer, prompt=200, completion=30)
        if kw.get("max_tokens") == 1:                      # the verifier
            return response("Yes", top=[{"token": "Yes", "logprob": math.log(0.9)},
                                        {"token": "No", "logprob": math.log(0.1)}],
                            prompt=80, completion=1)
        return response(self.answer, top=self.top, prompt=100, completion=10)

    def sample(self, messages, model, k=5, **kw):
        self.calls.append({"model": model, "sample_k": k})
        return [response(self.answer, prompt=100, completion=10) for _ in range(k)]


GSM = {"task_id": "g1", "dataset": "gsm8k", "query": "What is 6 times 7?",
       "gold_answer": "42", "tool_necessity": "not_required", "tool_type": "calculator"}


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------

def test_confident_answer_routes_direct_and_grades_correct():
    rec = run_task(StubClient(answer="42", top=CONFIDENT), GSM, RunConfig())
    assert rec.routing_decision == RoutingDecision.DIRECT
    assert rec.tool_called is False
    assert rec.tool_used == ToolType.NONE
    assert rec.correct is True
    assert rec.confidence_score is not None and rec.confidence_score > 0.9
    print("test_confident_answer_routes_direct_and_grades_correct: PASS")


def test_unsure_answer_routes_to_the_calculator():
    # two equally likely tokens -> confidence 0.0 -> below any threshold
    client = StubClient(answer="41", top=UNSURE, expression="6*7")
    rec = run_task(client, GSM, RunConfig())
    assert rec.confidence_score is not None and rec.confidence_score < 0.1
    assert rec.routing_decision == RoutingDecision.TOOL
    assert rec.tool_used == ToolType.CALCULATOR
    # the calculator evaluated 6*7 and corrected the model's 41
    assert rec.model_answer == "42"
    assert rec.correct is True
    assert rec.meta["tool_helped"] is True
    print("test_unsure_answer_routes_to_the_calculator: PASS")


def test_unparseable_expression_falls_back_to_the_direct_answer():
    # The tool could not help. That is an honest unnecessary tool call,
    # not a crash -- and the cost of trying is still billed.
    client = StubClient(answer="41", top=UNSURE, expression="I cannot compute this")
    rec = run_task(client, GSM, RunConfig())
    assert rec.tool_called is True
    assert rec.model_answer == "41"
    assert rec.correct is False
    assert rec.meta["tool_helped"] is False
    assert rec.tool_prompt_tokens > 0, "the failed tool call still costs money"
    print("test_unparseable_expression_falls_back_to_the_direct_answer: PASS")


# ---------------------------------------------------------------------------
# Cost accounting -- the reason the schema was widened
# ---------------------------------------------------------------------------

def test_entropy_costs_nothing_extra():
    # logprobs ride along on the answer call; there is no second call.
    rec = run_task(StubClient(top=CONFIDENT), GSM, RunConfig())
    assert rec.confidence_prompt_tokens == 0
    assert rec.confidence_completion_tokens == 0
    assert rec.confidence_api_cost_usd == 0.0
    print("test_entropy_costs_nothing_extra: PASS")


def test_self_consistency_bills_its_extra_samples():
    # k=5 stub samples at 100 prompt / 10 completion each
    client = StubClient(top=CONFIDENT)
    rec = run_task(client, GSM, RunConfig(method=ConfidenceMethod.SELF_CONSISTENCY, k=5))
    assert rec.confidence_prompt_tokens == 500, rec.confidence_prompt_tokens
    assert rec.confidence_completion_tokens == 50, rec.confidence_completion_tokens
    assert rec.confidence_model_name == rec.model_name
    print("test_self_consistency_bills_its_extra_samples: PASS")


def test_verifier_tokens_are_recorded_even_when_unpriced():
    # The open-weight verifier is not in PRICING_TABLE yet. Its tokens must
    # still be visible in the record rather than vanishing -- the warning
    # says they are excluded from CPST, and meta says how many they were.
    client = StubClient(top=CONFIDENT)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rec = run_task(client, GSM,
                       RunConfig(method=ConfidenceMethod.EXTERNAL_LLM,
                                 verifier_model="llama-3.1-8b"))
    assert rec.meta["verifier_prompt_tokens"] == 80, rec.meta
    assert rec.meta["verifier_model"] == "llama-3.1-8b"
    assert any("PRICING_TABLE" in str(w.message) for w in caught), \
        "an unpriced verifier must warn, not silently cost nothing"
    print("test_verifier_tokens_are_recorded_even_when_unpriced: PASS")


def test_verifier_is_priced_when_the_model_is_known():
    # gpt-4o-mini IS in the table: 80 prompt + 1 completion
    #   80 * 0.15/1M + 1 * 0.60/1M = 0.000012 + 0.0000006 = 0.0000126
    client = StubClient(top=CONFIDENT)
    rec = run_task(client, GSM,
                   RunConfig(method=ConfidenceMethod.EXTERNAL_LLM,
                             verifier_model="gpt-4o-mini"))
    assert abs(rec.confidence_api_cost_usd - 0.0000126) < 1e-12, rec.confidence_api_cost_usd
    print("test_verifier_is_priced_when_the_model_is_known: PASS")


def test_latency_is_split_by_stage():
    rec = run_task(StubClient(top=UNSURE), GSM, RunConfig())
    assert rec.latency.main_model_ms >= 0
    assert rec.latency.tool_ms >= 0
    assert rec.latency.total_ms >= rec.latency.main_model_ms
    print("test_latency_is_split_by_stage: PASS")


# ---------------------------------------------------------------------------
# Routing policy reaches the loop
# ---------------------------------------------------------------------------

def test_missing_confidence_escalates():
    # No logprobs at all -> entropy returns None -> the router's default
    # escalates rather than answering confidently from nothing.
    class NoLogprobs(StubClient):
        def complete(self, messages, model, **kw):
            if "ONE Python expression" in messages[0].get("content", ""):
                return response("6*7", prompt=40, completion=8)
            return {"choices": [{"message": {"content": "41"}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10}}

    rec = run_task(NoLogprobs(), GSM, RunConfig())
    assert rec.confidence_score is None
    assert rec.routing_decision == RoutingDecision.TOOL
    print("test_missing_confidence_escalates: PASS")


def test_on_missing_direct_flips_the_policy():
    class NoLogprobs(StubClient):
        def complete(self, messages, model, **kw):
            return {"choices": [{"message": {"content": "41"}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10}}

    rec = run_task(NoLogprobs(), GSM, RunConfig(on_missing="direct"))
    assert rec.routing_decision == RoutingDecision.DIRECT
    print("test_on_missing_direct_flips_the_policy: PASS")


# ---------------------------------------------------------------------------
# run_dataset
# ---------------------------------------------------------------------------

def test_failing_task_is_skipped_not_fatal():
    # One bad task should not cost you the other two in a paid run.
    class Flaky(StubClient):
        def complete(self, messages, model, **kw):
            if "boom" in messages[-1].get("content", ""):
                raise RuntimeError("provider exploded")
            return super().complete(messages, model, **kw)

    tasks = [dict(GSM, task_id="a"), dict(GSM, task_id="b", query="boom"),
             dict(GSM, task_id="c")]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        records = run_dataset(Flaky(top=CONFIDENT), tasks, RunConfig())
    assert [r.task_id for r in records] == ["a", "c"]
    assert any("failed and were skipped" in str(w.message) for w in caught)
    print("test_failing_task_is_skipped_not_fatal: PASS")


def test_on_error_raise_propagates():
    class Boom(StubClient):
        def complete(self, messages, model, **kw):
            raise RuntimeError("provider exploded")

    try:
        run_dataset(Boom(), [GSM], RunConfig(), on_error="raise")
    except RuntimeError:
        print("test_on_error_raise_propagates: PASS")
        return
    raise AssertionError("expected the error to propagate")


def test_rows_are_logged_as_they_complete():
    # Append-as-you-go, so a run that dies leaves a usable partial log.
    logged = []
    run_dataset(StubClient(top=CONFIDENT), [dict(GSM, task_id=f"t{i}") for i in range(3)],
                RunConfig(), logger=type("L", (), {"log": lambda s, r: logged.append(r)})())
    assert [r.task_id for r in logged] == ["t0", "t1", "t2"]
    print("test_rows_are_logged_as_they_complete: PASS")


def test_package_imports_without_the_openai_sdk():
    # The harness is stdlib-only except for the one module that makes a
    # network call, and client.py keeps it that way by importing openai
    # lazily inside Client.__post_init__. Moving that import to module
    # level would break every stdlib-only path -- including this whole
    # test suite, which never makes a call -- so it is pinned here.
    import builtins
    import importlib
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "openai" or name.startswith("openai."):
            raise ImportError("simulated: openai not installed")
        return real_import(name, *a, **k)

    builtins.__import__ = blocked
    try:
        for name in ("eval_harness.client", "eval_harness.runner", "eval_harness"):
            importlib.reload(importlib.import_module(name))
        from eval_harness.client import Client, MissingAPIKey
        try:
            Client(provider="cerebras", api_key="x")
        except ImportError as exc:
            assert "pip install" in str(exc), exc
        else:
            raise AssertionError("expected an ImportError naming the fix")
    finally:
        builtins.__import__ = real_import
        importlib.reload(importlib.import_module("eval_harness.client"))
        importlib.reload(importlib.import_module("eval_harness"))
    print("test_package_imports_without_the_openai_sdk: PASS")


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
