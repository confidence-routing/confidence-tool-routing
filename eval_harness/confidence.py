"""
confidence.py
=============

Confidence estimators. Each produces a single ``confidence_score`` in
[0, 1] suitable for ``TaskRecord.confidence_score``, paired with the
matching ``ConfidenceMethod`` tag.

Semantics (fixed by metrics.py): ``confidence_score`` is an estimate of
P(the model's DIRECT answer is correct). It is NOT P(a tool is needed).
compute_ece() bins this value against ``TaskRecord.correct``, so anything
we put here has to be calibrated against direct-answer correctness. The
router inverts it: low confidence -> call a tool.

Estimators (built in order):
    (a) token entropy      -- THIS FILE, below
    (b) self-consistency   -- next
    (c) external verifier  -- next
    (d) hybrid combiner    -- next

--- (a) Token-entropy estimator ---------------------------------------

Black-box by construction. We read only what the OpenAI chat completions
API returns to any paying caller -- ``logprobs`` / ``top_logprobs`` on the
sampled tokens -- and never touch hidden states or intermediate-layer
activations. This is the deliberate departure from MICE (Subramani et
al., 2025, arXiv:2504.20168), whose logit-lens features require
open-weight models; everything here works against a frozen API endpoint.

This module is pure: it takes an already-completed API response as input
and makes no network calls, so the scoring math is unit-testable without
a key.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .models import ConfidenceMethod

# Probabilities below this are treated as zero, both for the residual
# tail mass and for the 0*log(0) == 0 convention in the entropy sum.
_EPS = 1e-12

AGGREGATIONS = ("mean", "min", "first")


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------

def _get(obj: Any, key: str) -> Any:
    """Read `key` off either a dict or an object with attributes.

    The OpenAI SDK returns pydantic objects, but tests (and cached
    responses reloaded from JSON) hand us plain dicts. Supporting both
    means the estimator never depends on the openai package being
    installed.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def extract_token_logprobs(response: Any) -> List[Dict[str, Any]]:
    """
    Pull the per-token logprob entries out of a chat completion response.

    Accepts, in order of preference:
        - a full response: response.choices[0].logprobs.content
        - a logprobs payload: {"content": [...]}
        - the token list itself: [{"token": ..., "logprob": ..., "top_logprobs": [...]}, ...]

    Returns [] when the response carries no logprobs (e.g. the call was
    made without logprobs=True) rather than raising -- the caller decides
    whether a missing signal is fatal.
    """
    if response is None:
        return []

    # already the token list?
    if isinstance(response, (list, tuple)):
        return [t for t in response if t is not None]

    # a logprobs payload?
    content = _get(response, "content")
    if isinstance(content, (list, tuple)):
        return [t for t in content if t is not None]

    # a full response
    choices = _get(response, "choices")
    if isinstance(choices, (list, tuple)) and choices:
        logprobs = _get(choices[0], "logprobs")
        content = _get(logprobs, "content")
        if isinstance(content, (list, tuple)):
            return [t for t in content if t is not None]

    return []


def _candidate_probs(token_entry: Any) -> List[float]:
    """
    Probabilities of the candidate tokens at one position.

    Prefers ``top_logprobs`` (the top-k alternatives). Falls back to the
    sampled token's own ``logprob`` as a single candidate when
    top_logprobs is absent -- that still carries real signal (p of the
    chosen token vs. everything else), just coarser.
    """
    top = _get(token_entry, "top_logprobs")
    if isinstance(top, (list, tuple)) and top:
        probs = []
        for cand in top:
            lp = _get(cand, "logprob")
            if lp is not None:
                probs.append(math.exp(float(lp)))
        if probs:
            return probs

    lp = _get(token_entry, "logprob")
    if lp is not None:
        return [math.exp(float(lp))]

    return []


# ---------------------------------------------------------------------------
# Entropy math
# ---------------------------------------------------------------------------

def normalized_token_entropy(probs: Sequence[float], renormalize: bool = False) -> Optional[float]:
    """
    Normalized Shannon entropy of one token position, in [0, 1].

    The API gives us a TRUNCATED distribution: top_logprobs=k returns the
    k most likely tokens, whose probabilities sum to less than 1. Two
    defensible ways to handle the missing mass, both supported:

    renormalize=False (default, "lump the tail"):
        Treat the leftover mass (1 - sum p_i) as one additional outcome.
        This is the more honest reading -- the tail exists and it is
        uncertainty -- and it degrades gracefully at k=1, where the
        renormalize path would divide by log(1) = 0.

    renormalize=True:
        Discard the tail and rescale the top-k to sum to 1. Comparable to
        how truncated-entropy is usually reported, but it understates
        uncertainty on flat distributions where most mass fell outside k.

    Normalizing by log(n_eff) -- where n_eff counts outcomes carrying
    non-negligible mass -- makes the score independent of the top_logprobs
    setting, so entropy confidences computed with k=5 and k=20 are on the
    same scale. That comparability is what lets us pool calibration
    numbers across the four tool categories.

    Returns None if there are no candidate probabilities at all.
    A single certain outcome (n_eff == 1) has zero entropy -> returns 0.0.
    """
    probs = [max(0.0, float(p)) for p in probs]
    if not probs:
        return None

    total = sum(probs)

    if renormalize:
        if total <= _EPS:
            return None
        dist = [p / total for p in probs]
    else:
        residual = 1.0 - total
        dist = list(probs)
        if residual > _EPS:
            dist.append(residual)
        # total > 1 can happen from float error on a saturated top-1;
        # clamp so the distribution stays a distribution.
        s = sum(dist)
        if s > _EPS:
            dist = [p / s for p in dist]

    n_eff = sum(1 for p in dist if p > _EPS)
    if n_eff <= 1:
        return 0.0  # all mass on one outcome: no uncertainty

    entropy = -sum(p * math.log(p) for p in dist if p > _EPS)
    max_entropy = math.log(n_eff)

    return min(1.0, max(0.0, entropy / max_entropy))


# ---------------------------------------------------------------------------
# Public estimator
# ---------------------------------------------------------------------------

@dataclass
class EntropyConfidenceResult:
    """Full entropy trace for one completion.

    ``confidence`` is the number that goes into TaskRecord; the rest is
    kept for per-token analysis in the paper and for debugging a router
    decision after the fact.
    """
    confidence: Optional[float]
    mean_normalized_entropy: Optional[float]
    per_token_entropy: List[float] = field(default_factory=list)
    n_tokens_scored: int = 0
    n_tokens_skipped: int = 0
    aggregation: str = "mean"
    method: ConfidenceMethod = ConfidenceMethod.ENTROPY


def estimate_entropy_confidence(
    response: Any,
    *,
    aggregation: str = "mean",
    renormalize: bool = False,
    skip_whitespace: bool = False,
) -> EntropyConfidenceResult:
    """
    Confidence from token-level entropy of an OpenAI chat completion that
    was created with ``logprobs=True, top_logprobs=k``.

        confidence = 1 - aggregate(normalized per-token entropy)

    aggregation:
        "mean"  -- average uncertainty across the answer (default).
        "min"   -- confidence of the WORST token (i.e. 1 - max entropy).
                   Conservative: one wobbly token drags the whole answer
                   down. Useful when a single wrong token (a digit, an
                   entity name) invalidates the answer.
        "first" -- first token only. Cheap proxy that works when the
                   answer is effectively a single decision, e.g. a
                   yes/no or a multiple-choice letter.

    Returns a result whose ``confidence`` is None when the response
    carries no usable logprobs -- a missing signal, which compute_ece()
    drops, not a confident-wrong 0.0.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")

    tokens = extract_token_logprobs(response)

    per_token: List[float] = []
    n_skipped = 0
    for entry in tokens:
        if skip_whitespace:
            tok = _get(entry, "token")
            if isinstance(tok, str) and tok.strip() == "":
                n_skipped += 1
                continue

        h = normalized_token_entropy(_candidate_probs(entry), renormalize=renormalize)
        if h is None:
            n_skipped += 1
            continue
        per_token.append(h)

        if aggregation == "first":
            break

    if not per_token:
        return EntropyConfidenceResult(
            confidence=None,
            mean_normalized_entropy=None,
            per_token_entropy=[],
            n_tokens_scored=0,
            n_tokens_skipped=n_skipped,
            aggregation=aggregation,
        )

    mean_entropy = sum(per_token) / len(per_token)

    if aggregation == "mean":
        aggregate_entropy = mean_entropy
    elif aggregation == "min":
        aggregate_entropy = max(per_token)   # worst token -> lowest confidence
    else:  # "first"
        aggregate_entropy = per_token[0]

    confidence = min(1.0, max(0.0, 1.0 - aggregate_entropy))

    return EntropyConfidenceResult(
        confidence=confidence,
        mean_normalized_entropy=mean_entropy,
        per_token_entropy=per_token,
        n_tokens_scored=len(per_token),
        n_tokens_skipped=n_skipped,
        aggregation=aggregation,
    )


def token_entropy_confidence(response: Any, **kwargs: Any) -> Optional[float]:
    """Just the scalar, for callers that only need TaskRecord.confidence_score."""
    return estimate_entropy_confidence(response, **kwargs).confidence
