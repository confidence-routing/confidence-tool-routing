"""
costs.py
========

OpenAI token pricing and cost-estimation helpers.

IMPORTANT: OpenAI revises pricing periodically. The table below reflects
rates commonly reported as of mid-2026. Before running real experiments,
verify against https://openai.com/api/pricing and update PRICING_TABLE
and PRICING_VERIFIED_DATE. Keeping this in one place means a price change
is a one-line edit, not a hunt through the codebase.

All rates are USD per 1,000,000 tokens.
"""

from __future__ import annotations

import datetime
import warnings
from typing import Dict, NamedTuple

# ------------------------------------------------------------------
# Staleness guard
# ------------------------------------------------------------------
# Update this date every time you verify PRICING_TABLE against
# https://openai.com/api/pricing.  estimate_cost_usd() will emit a
# warning if the table is more than STALENESS_THRESHOLD_DAYS old,
# so you never silently publish CPST numbers based on stale rates.
# ------------------------------------------------------------------
PRICING_VERIFIED_DATE: datetime.date = datetime.date(2026, 8, 20)
STALENESS_THRESHOLD_DAYS: int = 30

_staleness_warned: bool = False


def _check_pricing_staleness() -> None:
    """Warn once per process if the pricing table hasn't been verified
    within STALENESS_THRESHOLD_DAYS."""
    global _staleness_warned
    if _staleness_warned:
        return
    age_days = (datetime.date.today() - PRICING_VERIFIED_DATE).days
    if age_days > STALENESS_THRESHOLD_DAYS:
        _staleness_warned = True
        warnings.warn(
            f"PRICING_TABLE was last verified {age_days} days ago "
            f"({PRICING_VERIFIED_DATE.isoformat()}). CPST numbers may be "
            f"inaccurate. Verify rates at https://openai.com/api/pricing "
            f"and update PRICING_VERIFIED_DATE in costs.py.",
            UserWarning,
            stacklevel=3,
        )


class ModelPricing(NamedTuple):
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float = 0.0  # 0.0 = not cached / not applicable


PRICING_TABLE: Dict[str, ModelPricing] = {
    # main-model candidates
    "gpt-4o":            ModelPricing(2.50, 10.00, 1.25),
    "gpt-4.1":            ModelPricing(2.00, 8.00, 0.50),
    "gpt-4.1-mini":       ModelPricing(0.40, 1.60, 0.10),
    "gpt-4.1-nano":       ModelPricing(0.10, 0.40, 0.025),
    "gpt-5":              ModelPricing(1.25, 10.00, 0.125),
    "gpt-5-mini":         ModelPricing(0.25, 2.00, 0.025),

    # lightweight external-verifier candidates
    "gpt-4o-mini":        ModelPricing(0.15, 0.60, 0.075),
    "o4-mini":            ModelPricing(1.10, 4.40, 0.275),
}

DEFAULT_MODEL = "gpt-4o-mini"


def estimate_cost_usd(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
) -> float:
    """
    Estimate USD cost for a single call.

    cached_prompt_tokens: tokens served from OpenAI's prompt cache, billed
    at the discounted cached rate. Must be <= prompt_tokens; the remainder
    of prompt_tokens is billed at the standard input rate.
    """
    _check_pricing_staleness()

    if model_name not in PRICING_TABLE:
        raise KeyError(
            f"No pricing entry for '{model_name}'. "
            f"Add it to PRICING_TABLE in costs.py (check openai.com/api/pricing)."
        )
    pricing = PRICING_TABLE[model_name]

    cached_prompt_tokens = min(cached_prompt_tokens, prompt_tokens)
    uncached_prompt_tokens = prompt_tokens - cached_prompt_tokens

    cost = (
        uncached_prompt_tokens * pricing.input_per_million
        + cached_prompt_tokens * pricing.cached_input_per_million
        + completion_tokens * pricing.output_per_million
    ) / 1_000_000

    return cost


def estimate_record_cost_usd(record) -> float:
    """
    Convenience wrapper: estimate total USD cost for a TaskRecord.

    Three stages, all billed:

      main        the answer itself, at record.model_name
      tool        the tool round-trip's extra tokens, also at
                  record.model_name, plus flat tool_api_cost_usd for
                  non-token charges (e.g. a paid web-search call)
      confidence  tokens spent ESTIMATING confidence rather than
                  answering -- self-consistency's k-1 extra samples, the
                  external verifier's call -- at confidence_model_name
                  when set, else record.model_name, plus flat
                  confidence_api_cost_usd

    The confidence stage is billed separately because it is routinely a
    different model: a verifier is deliberately cheaper than the model it
    checks. Leaving it out (as this function did before) makes free token
    entropy and k-sample self-consistency cost exactly the same, which
    hides the one difference the routing argument depends on.

    Records logged before these fields existed default every one of them
    to zero, so old runs price exactly as they did before.
    """
    main_cost = estimate_cost_usd(
        record.model_name, record.prompt_tokens, record.completion_tokens
    )

    tool_token_cost = 0.0
    if record.tool_prompt_tokens or record.tool_completion_tokens:
        tool_token_cost = estimate_cost_usd(
            record.model_name, record.tool_prompt_tokens, record.tool_completion_tokens
        )

    confidence_token_cost = 0.0
    if record.confidence_prompt_tokens or record.confidence_completion_tokens:
        confidence_token_cost = estimate_cost_usd(
            record.confidence_model_name or record.model_name,
            record.confidence_prompt_tokens,
            record.confidence_completion_tokens,
        )

    return (
        main_cost
        + tool_token_cost
        + record.tool_api_cost_usd
        + confidence_token_cost
        + record.confidence_api_cost_usd
    )
