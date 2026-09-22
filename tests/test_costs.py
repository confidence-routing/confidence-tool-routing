"""
test_costs.py
=============

Unit tests for eval_harness.costs, focused on the pricing-table staleness
guard added alongside estimate_cost_usd().

The guard reads two module-level globals -- PRICING_VERIFIED_DATE and the
_staleness_warned latch -- so these tests drive it by rewriting those
globals rather than by faking the clock. Setting PRICING_VERIFIED_DATE
relative to the real date.today() keeps every case deterministic without
pulling in a freezegun-style dependency (this project is stdlib-only).

Run:
    python -m pytest tests/ -v
    (or, without pytest installed: python -m tests.test_costs)
"""

import datetime
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness import costs
from eval_harness.costs import estimate_cost_usd, estimate_record_cost_usd
from eval_harness.models import TaskRecord


@contextmanager
def pricing_age(days: int):
    """Pretend PRICING_TABLE was last verified `days` days ago, with the
    once-per-process warning latch reset. Restores both globals after."""
    saved_date = costs.PRICING_VERIFIED_DATE
    saved_latch = costs._staleness_warned
    costs.PRICING_VERIFIED_DATE = datetime.date.today() - datetime.timedelta(days=days)
    costs._staleness_warned = False
    try:
        yield
    finally:
        costs.PRICING_VERIFIED_DATE = saved_date
        costs._staleness_warned = saved_latch


@contextmanager
def captured_warnings():
    """Collect every warning raised in the block, defeating the default
    'once per location' filter so repeat-call behaviour is observable."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


# ---------------------------------------------------------------------------
# Staleness guard
# ---------------------------------------------------------------------------

def test_fresh_pricing_table_does_not_warn():
    # verified 1 day ago, threshold is 30 -> well inside the window
    with pricing_age(1), captured_warnings() as caught:
        estimate_cost_usd("gpt-4o-mini", 1000, 1000)
    assert caught == [], f"expected no warning, got {[str(w.message) for w in caught]}"
    print("test_fresh_pricing_table_does_not_warn: PASS")


def test_stale_pricing_table_warns():
    # verified 31 days ago -> 31 > 30 -> warn
    with pricing_age(costs.STALENESS_THRESHOLD_DAYS + 1), captured_warnings() as caught:
        estimate_cost_usd("gpt-4o-mini", 1000, 1000)
    assert len(caught) == 1, f"expected exactly 1 warning, got {len(caught)}"
    assert issubclass(caught[0].category, UserWarning)
    message = str(caught[0].message)
    # the message has to be actionable: how stale, and what to edit
    assert "31 days ago" in message
    assert "PRICING_VERIFIED_DATE" in message
    assert "openai.com/api/pricing" in message
    print("test_stale_pricing_table_warns: PASS")


def test_staleness_boundary_is_exclusive():
    # exactly at the threshold is still considered fresh: the guard fires on
    # age_days > STALENESS_THRESHOLD_DAYS, not >=
    with pricing_age(costs.STALENESS_THRESHOLD_DAYS), captured_warnings() as caught:
        estimate_cost_usd("gpt-4o-mini", 1000, 1000)
    assert caught == [], "threshold day itself should not warn"
    print("test_staleness_boundary_is_exclusive: PASS")


def test_stale_warning_fires_only_once_per_process():
    # a sweep over hundreds of records should not emit hundreds of warnings
    with pricing_age(365), captured_warnings() as caught:
        for _ in range(5):
            estimate_cost_usd("gpt-4o-mini", 1000, 1000)
    assert len(caught) == 1, f"expected 1 warning across 5 calls, got {len(caught)}"
    print("test_stale_warning_fires_only_once_per_process: PASS")


def test_staleness_guard_runs_before_unknown_model_error():
    # the guard sits at the top of estimate_cost_usd, so a stale table still
    # warns even when the call goes on to fail on an unknown model name
    with pricing_age(365), captured_warnings() as caught:
        try:
            estimate_cost_usd("not-a-real-model", 1000, 1000)
        except KeyError:
            pass
        else:
            raise AssertionError("expected KeyError for unknown model")
    assert len(caught) == 1
    print("test_staleness_guard_runs_before_unknown_model_error: PASS")


def test_shipped_verified_date_is_not_in_the_future():
    # cheap typo guard on the committed constant: a future date would silence
    # the staleness check indefinitely (age_days goes negative)
    assert costs.PRICING_VERIFIED_DATE <= datetime.date.today(), (
        f"PRICING_VERIFIED_DATE {costs.PRICING_VERIFIED_DATE.isoformat()} is in "
        f"the future, which disables the staleness guard"
    )
    print("test_shipped_verified_date_is_not_in_the_future: PASS")


# ---------------------------------------------------------------------------
# Cost arithmetic (guard must not perturb the numbers)
# ---------------------------------------------------------------------------

def test_cached_tokens_billed_at_discounted_rate():
    # gpt-4o-mini: 0.15 in / 0.60 out / 0.075 cached, per 1M tokens.
    # 1M prompt tokens with 400k cached, 0 completion:
    #   600_000 * 0.15/1M + 400_000 * 0.075/1M = 0.09 + 0.03 = 0.12
    with pricing_age(1):
        cost = estimate_cost_usd("gpt-4o-mini", 1_000_000, 0, cached_prompt_tokens=400_000)
    assert abs(cost - 0.12) < 1e-9, cost
    print("test_cached_tokens_billed_at_discounted_rate: PASS")


def test_cached_tokens_clamped_to_prompt_tokens():
    # over-reporting cached tokens must not produce a cheaper-than-possible
    # bill: 1000 prompt tokens all cached -> 1000 * 0.075/1M
    with pricing_age(1):
        cost = estimate_cost_usd("gpt-4o-mini", 1000, 0, cached_prompt_tokens=999_999)
    assert abs(cost - (1000 * 0.075 / 1_000_000)) < 1e-12, cost
    print("test_cached_tokens_clamped_to_prompt_tokens: PASS")


# ---------------------------------------------------------------------------
# Confidence-stage billing
#
# The confidence stage costs real money -- self-consistency resamples k
# times, the external verifier makes its own call -- and until these fields
# existed estimate_record_cost_usd() charged for none of it. That made free
# token entropy and 5-sample self-consistency price identically, hiding the
# exact difference the routing argument rests on.
#
# gpt-4o:      2.50 in / 10.00 out per 1M tokens
# gpt-4o-mini: 0.15 in /  0.60 out per 1M tokens
# ---------------------------------------------------------------------------

def _rec(**kw):
    base = dict(task_id="t", dataset="d", model_name="gpt-4o",
                prompt_tokens=1000, completion_tokens=100)
    base.update(kw)
    return TaskRecord(**base)


# main answer, used as the baseline by several tests below:
#   1000 * 2.50/1M + 100 * 10.00/1M = 0.0025 + 0.0010 = 0.0035
_MAIN_ONLY = 0.0035


def test_confidence_tokens_are_billed():
    # confidence stage on the same model as the answer (no override):
    #   500 * 2.50/1M + 1 * 10.00/1M = 0.00125 + 0.00001 = 0.00126
    # total = 0.0035 + 0.00126 = 0.00476
    with pricing_age(1):
        cost = estimate_record_cost_usd(
            _rec(confidence_prompt_tokens=500, confidence_completion_tokens=1)
        )
    assert abs(cost - 0.00476) < 1e-9, cost
    print("test_confidence_tokens_are_billed: PASS")


def test_confidence_uses_its_own_model_rate():
    # same 500+1 tokens, but on the cheap verifier:
    #   500 * 0.15/1M + 1 * 0.60/1M = 0.000075 + 0.0000006 = 0.0000756
    # total = 0.0035 + 0.0000756 = 0.0035756
    #
    # This is the whole reason confidence_model_name exists: a verifier is
    # deliberately cheaper than the model it checks, and pricing it at the
    # main model's rate would overstate it ~17x here.
    with pricing_age(1):
        cost = estimate_record_cost_usd(
            _rec(confidence_prompt_tokens=500, confidence_completion_tokens=1,
                 confidence_model_name="gpt-4o-mini")
        )
    assert abs(cost - 0.0035756) < 1e-9, cost
    print("test_confidence_uses_its_own_model_rate: PASS")


def test_blank_confidence_model_falls_back_to_main_model():
    # "" must mean "same model as the answer", not "no model" -- the two
    # records below differ only in whether the field is spelled out.
    with pricing_age(1):
        implicit = estimate_record_cost_usd(
            _rec(confidence_prompt_tokens=500, confidence_completion_tokens=1))
        explicit = estimate_record_cost_usd(
            _rec(confidence_prompt_tokens=500, confidence_completion_tokens=1,
                 confidence_model_name="gpt-4o"))
    assert abs(implicit - explicit) < 1e-12, (implicit, explicit)
    print("test_blank_confidence_model_falls_back_to_main_model: PASS")


def test_flat_confidence_api_cost_is_added():
    # escape hatch for a confidence stage spanning two models: 0.0035 + 0.002
    with pricing_age(1):
        cost = estimate_record_cost_usd(_rec(confidence_api_cost_usd=0.002))
    assert abs(cost - 0.0055) < 1e-9, cost
    print("test_flat_confidence_api_cost_is_added: PASS")


def test_records_without_confidence_fields_price_unchanged():
    # Runs logged before these fields existed must price exactly as before,
    # or every historical CPST number silently shifts.
    with pricing_age(1):
        cost = estimate_record_cost_usd(_rec())
    assert abs(cost - _MAIN_ONLY) < 1e-12, cost
    print("test_records_without_confidence_fields_price_unchanged: PASS")


def test_old_log_rows_still_deserialize():
    # from_dict on a row written before the fields existed: the dataclass
    # defaults fill in, so replaying an old run costs what it always did.
    row = {"task_id": "t", "dataset": "d", "model_name": "gpt-4o",
           "prompt_tokens": 1000, "completion_tokens": 100}
    rec = TaskRecord.from_dict(row)
    assert rec.confidence_prompt_tokens == 0
    assert rec.confidence_completion_tokens == 0
    assert rec.confidence_model_name == ""
    assert rec.confidence_api_cost_usd == 0.0
    with pricing_age(1):
        assert abs(estimate_record_cost_usd(rec) - _MAIN_ONLY) < 1e-12
    print("test_old_log_rows_still_deserialize: PASS")


def test_token_totals_include_confidence_stage():
    # token-denominated CPST has to move with the dollar one, or the two
    # headline numbers disagree about what a run cost.
    rec = _rec(tool_prompt_tokens=10, tool_completion_tokens=20,
               confidence_prompt_tokens=500, confidence_completion_tokens=1)
    assert rec.total_prompt_tokens == 1000 + 10 + 500, rec.total_prompt_tokens
    assert rec.total_completion_tokens == 100 + 20 + 1, rec.total_completion_tokens
    print("test_token_totals_include_confidence_stage: PASS")


def test_self_consistency_costs_more_than_entropy():
    # The point of the whole change, stated as an assertion: token entropy
    # is free (logprobs ride along on the answer call), 5-sample
    # self-consistency pays for 4 extra answers. Before this fix these two
    # records priced identically.
    with pricing_age(1):
        entropy = estimate_record_cost_usd(_rec())
        self_consistency = estimate_record_cost_usd(
            _rec(confidence_prompt_tokens=4 * 1000,
                 confidence_completion_tokens=4 * 100))
    assert abs(entropy - _MAIN_ONLY) < 1e-12, entropy
    assert abs(self_consistency - 5 * _MAIN_ONLY) < 1e-9, self_consistency
    assert self_consistency > entropy
    print("test_self_consistency_costs_more_than_entropy: PASS")


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
