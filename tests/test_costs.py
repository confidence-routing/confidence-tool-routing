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
from eval_harness.costs import estimate_cost_usd


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


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
