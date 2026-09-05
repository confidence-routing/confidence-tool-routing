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
    (b) self-consistency   -- THIS FILE, below
    (c) external verifier  -- THIS FILE, below
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

--- (b) Self-consistency estimator -------------------------------------

Sample the same query k times at non-zero temperature and measure how
often the samples agree (Wang et al., 2023, arXiv:2203.11171 -- there
used to pick an answer by majority vote; here the vote MARGIN is the
signal, not the winner). Agreement rate is a natural estimate of
P(direct answer correct): a model that lands on the same answer 9 times
out of 10 is usually right, and one that scatters usually is not.

Complementary to (a) rather than redundant with it. Token entropy reads
uncertainty inside a single forward pass and is nearly free (one call,
logprobs are metadata). Self-consistency reads uncertainty across
independent samples and costs k times as much -- but it catches the case
entropy misses, where the model is fluent and locally certain on every
token of an answer it would not reproduce given another roll. The cost
asymmetry is the point of the routing experiment: entropy is cheap
enough to run on every task, self-consistency is what you escalate to.

Also pure -- takes already-collected samples, makes no network calls.

--- (c) External-verifier estimator ------------------------------------

Ask a cheap second model whether the main model's answer is right, and
turn its judgement into a confidence. One extra call to a small model
(gpt-4o-mini, o4-mini) against one call to the big one, so it sits
between (a) and (b) on cost: dearer than reading logprobs you already
have, far cheaper than k full resamples of an expensive model.

The signal is different in kind from both. (a) and (b) ask the model
about itself and inherit its blind spots -- a model confidently wrong
about a fact is fluent AND reproducible, so entropy is low and agreement
is high, and both estimators say "confident". A separate model with
different training data has no stake in the first model's answer, which
is the one way in this family to catch confidently-wrong-and-consistent.

Scoring prefers the yes-token PROBABILITY over the verifier's verbalized
score. Verbalized confidences from LLMs are badly calibrated and pile up
on round numbers (0.8, 0.9, 0.95); P(yes) read off top_logprobs is
continuous and far better behaved. That is why parsing prefers logprobs
and falls back to text only when logprobs were not requested -- and why
callers should send logprobs=True, top_logprobs>=5 on verifier calls.

Pure like the others: prompt construction and response parsing are
separate functions, so this module still makes no network calls. The
caller owns the API call in between.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .models import ConfidenceMethod

# Probabilities below this are treated as zero, both for the residual
# tail mass and for the 0*log(0) == 0 convention in the entropy sum.
_EPS = 1e-12

AGGREGATIONS = ("mean", "min", "first")

# Scoring rules for the self-consistency estimator, see
# estimate_self_consistency_confidence().
SC_SCORINGS = ("agreement", "entropy")

# Parsing rules for the external-verifier estimator, see
# estimate_external_verifier_confidence().
VERIFIER_SCORINGS = ("auto", "logprob", "verbalized", "binary")


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


# ---------------------------------------------------------------------------
# (b) Self-consistency
# ---------------------------------------------------------------------------

# Trailing punctuation carries no answer content ("42." == "42"), but a
# closing paren can ("f(x)"), so only strip what is unambiguously terminal.
_TRAILING_PUNCT = ".,;:!?'\"`"

# Prefixes models habitually prepend to a final answer. Stripped so
# "The answer is 42" and a bare "42" land in the same vote bucket.
_ANSWER_PREFIXES = (
    "the answer is",
    "answer:",
    "answer is",
    "final answer:",
    "final answer is",
)


def normalize_answer(text: Any) -> Optional[str]:
    """
    Canonical form of a free-text answer, for vote bucketing.

    Two samples that mean the same thing have to hash to the same bucket
    or the agreement rate understates confidence -- an over-strict
    comparison makes every model look uncalibrated. Handles the cheap,
    safe cases only: case, surrounding whitespace, a boilerplate answer
    prefix, terminal punctuation, and numeric spelling ("1,234.00",
    "$1234", "1234.0" -> "1234").

    Deliberately NOT handled: semantic equivalence ("NYC" vs "New York
    City"), unit conversion, or set/order equivalence. Those need a
    task-specific grader; pass your own via the ``normalizer`` argument
    when the dataset needs it.

    Returns None for input that carries no answer at all (None, or blank
    after normalization), which the estimator counts as a skipped sample
    rather than a vote for "".
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)

    out = " ".join(text.strip().lower().split())
    if not out:
        return None

    for prefix in _ANSWER_PREFIXES:
        if out.startswith(prefix):
            out = out[len(prefix):].strip()
            break

    out = out.strip(_TRAILING_PUNCT).strip()
    if not out:
        return None

    numeric = _canonical_number(out)
    return numeric if numeric is not None else out


def _canonical_number(text: str) -> Optional[str]:
    """Canonical string for a numeric answer, or None if not a number.

    "$1,234.00", "1234", and "1234.0" are the same answer; string
    equality says they are three. Integral floats render without the
    ".0" so they collide with the integer spelling.
    """
    candidate = text.replace(",", "").replace("$", "").replace("%", "").strip()
    if not candidate or not re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)", candidate):
        return None
    try:
        value = float(candidate)
    except ValueError:
        return None
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def extract_sampled_answers(responses: Any) -> List[Optional[str]]:
    """
    Pull the answer text out of k sampled completions.

    Accepts, in order of preference:
        - a list of plain strings (already-extracted answers)
        - one response sampled with n=k: response.choices[*].message.content
        - a list of k separate responses, each with one choice

    The n=k and k-separate-calls shapes both appear in practice: n=k is
    one request and cheaper on prompt tokens, separate calls are what you
    need when the sampling parameters differ per sample. Returns one
    entry per sample, None where a sample carried no text, so the caller
    can tell "5 samples, 1 empty" from "4 samples".
    """
    if responses is None:
        return []

    if isinstance(responses, str):
        return [responses]

    if isinstance(responses, (list, tuple)):
        # list of strings, or list of per-call responses
        out: List[Optional[str]] = []
        for item in responses:
            if item is None or isinstance(item, str):
                out.append(item)
            else:
                out.extend(_answers_from_choices(item))
        return out

    return _answers_from_choices(responses)


def _answers_from_choices(response: Any) -> List[Optional[str]]:
    """Answer text for every choice on one response object."""
    choices = _get(response, "choices")
    if not isinstance(choices, (list, tuple)):
        return []
    out: List[Optional[str]] = []
    for choice in choices:
        message = _get(choice, "message")
        content = _get(message, "content")
        if content is None:
            content = _get(choice, "text")  # legacy completions shape
        out.append(content if isinstance(content, str) else None)
    return out


def _normalized_vote_entropy(counts: Sequence[int], n_samples: int) -> float:
    """Shannon entropy of a vote distribution, normalized to [0, 1].

    Normalized by log(n_samples), NOT by log(number of distinct answers)
    the way normalized_token_entropy() does it. The denominators disagree
    and only this one ranks vote spread correctly.

    Take 10 samples. A clean 5-5 split against one rival and a 5 vs. five
    singletons scatter are both 0.5 agreement, and the scatter is plainly
    the less trustworthy of the two. Dividing by log(distinct answers)
    scores the tie 0.0 and the scatter 0.164 -- backwards, because a
    2-outcome uniform split is maximally uniform *among the outcomes
    present*. Dividing by log(n_samples) measures against the true
    worst case, k samples all disagreeing, and scores them 0.699 and
    0.349 respectively.

    That worst case is why normalized_token_entropy() is not reused here:
    for top_logprobs the outcome count is a sampling artifact of k, which
    is exactly what it has to normalize away; for votes the outcome count
    IS the signal.
    """
    if n_samples <= 1:
        return 0.0  # one sample cannot disagree with itself
    entropy = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / n_samples
        entropy -= p * math.log(p)
    return min(1.0, max(0.0, entropy / math.log(n_samples)))


@dataclass
class SelfConsistencyResult:
    """Full vote trace for one query sampled k times.

    ``confidence`` is the number that goes into TaskRecord; the rest is
    kept for the paper's per-task analysis and for explaining a router
    decision after the fact ("routed to tool: 3/8 agreement, top answers
    42 / 43 / 44").
    """
    confidence: Optional[float]
    consensus_answer: Optional[str]
    agreement_rate: Optional[float]
    vote_counts: List[Tuple[str, int]] = field(default_factory=list)
    n_samples: int = 0
    n_skipped: int = 0
    scoring: str = "agreement"
    method: ConfidenceMethod = ConfidenceMethod.SELF_CONSISTENCY


def estimate_self_consistency_confidence(
    responses: Any,
    *,
    scoring: str = "agreement",
    normalizer: Optional[Callable[[Any], Optional[str]]] = None,
) -> SelfConsistencyResult:
    """
    Confidence from agreement across k sampled answers to the same query.

    scoring:
        "agreement" -- (votes for the modal answer) / (valid samples).
                       Bounded [1/k, 1], directly interpretable as "the
                       model reproduces this answer x% of the time".
        "entropy"   -- 1 - normalized entropy of the full vote
                       distribution. Distinguishes cases "agreement"
                       flattens: 5/10 with one rival (5-5) and 5/10
                       scattered across five rivals both score 0.5 on
                       agreement, but the scattered one is the more
                       uncertain and scores lower here.

    normalizer: callable mapping a raw sample to a comparison key, or
        None for normalize_answer(). Return None from it to discard a
        sample. Pass a task-specific grader when string normalization is
        too weak (numeric tolerance, set equality, alias tables).

    ``consensus_answer`` is the winning normalized key -- the majority
    vote itself, which the caller can use as the answer as well as the
    confidence. Ties are broken by first appearance so the result is
    deterministic given the sample order.

    Returns confidence=None when no sample carried a usable answer -- a
    missing signal, which compute_ece() drops, not a confident-wrong 0.0.

    Note k=1 scores 1.0 under both rules: a single sample always agrees
    with itself. Self-consistency carries no information at k=1, so the
    caller is responsible for sampling k>1 -- an honest floor is not
    something this function can recover after the fact.
    """
    if scoring not in SC_SCORINGS:
        raise ValueError(f"scoring must be one of {SC_SCORINGS}, got {scoring!r}")

    normalize = normalizer or normalize_answer
    raw_samples = extract_sampled_answers(responses)

    keys: List[str] = []
    n_skipped = 0
    for sample in raw_samples:
        key = normalize(sample)
        if key is None:
            n_skipped += 1
            continue
        keys.append(key)

    if not keys:
        return SelfConsistencyResult(
            confidence=None,
            consensus_answer=None,
            agreement_rate=None,
            vote_counts=[],
            n_samples=0,
            n_skipped=n_skipped,
            scoring=scoring,
        )

    # Counter.most_common() does not promise a tie order; sort explicitly
    # by (-votes, first appearance) so equal counts resolve deterministically.
    counts = Counter(keys)
    first_seen = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))

    consensus, top_votes = ordered[0]
    agreement = top_votes / len(keys)

    if scoring == "agreement":
        confidence: Optional[float] = agreement
    else:  # "entropy"
        confidence = 1.0 - _normalized_vote_entropy(
            [c for _, c in ordered], len(keys)
        )

    if confidence is not None:
        confidence = min(1.0, max(0.0, confidence))

    return SelfConsistencyResult(
        confidence=confidence,
        consensus_answer=consensus,
        agreement_rate=agreement,
        vote_counts=ordered,
        n_samples=len(keys),
        n_skipped=n_skipped,
        scoring=scoring,
    )


def self_consistency_confidence(responses: Any, **kwargs: Any) -> Optional[float]:
    """Just the scalar, for callers that only need TaskRecord.confidence_score."""
    return estimate_self_consistency_confidence(responses, **kwargs).confidence


# ---------------------------------------------------------------------------
# (c) External lightweight verifier
# ---------------------------------------------------------------------------

# Surface forms of the verdict tokens. Matched after lowercasing and
# stripping whitespace/punctuation, so "Yes", " yes", and "yes." all hit.
_YES_TOKENS = frozenset({"yes", "y", "true", "correct", "right", "valid"})
_NO_TOKENS = frozenset({"no", "n", "false", "incorrect", "wrong", "invalid"})

VERIFIER_SYSTEM_PROMPT = (
    "You are a strict answer verifier. You will be shown a question and a "
    "candidate answer produced by another model. Decide whether the "
    "candidate answer is correct.\n"
    "Reply with exactly one word: Yes or No. No explanation, no punctuation."
)


def build_verifier_messages(
    query: str,
    answer: Optional[str],
    *,
    gold_answer: Optional[str] = None,
    system_prompt: str = VERIFIER_SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    """
    Chat messages asking a cheap model to verify `answer` to `query`.

    Constrained to a single Yes/No token on purpose. A one-token verdict
    is what makes the logprob read possible -- P(yes) at position 0 is
    the calibrated signal, and letting the verifier ramble first buries
    that token somewhere unpredictable in the stream. It also keeps the
    verifier's completion cost at one token.

    gold_answer is for OFFLINE ANALYSIS ONLY -- measuring the ceiling of
    a perfect verifier when scoring a cached dataset. Passing it during a
    live run leaks the label into the routing decision and invalidates
    the experiment. It is separately labelled in the prompt so a leak is
    obvious in a logged trace rather than silent.
    """
    user = f"Question:\n{query}\n\nCandidate answer:\n{answer if answer is not None else '(no answer given)'}"
    if gold_answer is not None:
        user += f"\n\nReference answer (ORACLE -- offline analysis only):\n{gold_answer}"
    user += "\n\nIs the candidate answer correct? Answer Yes or No."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def _verdict_key(token: Any) -> Optional[str]:
    """Map a raw token to "yes"/"no", or None if it is neither."""
    if not isinstance(token, str):
        return None
    key = token.strip().lower().strip(_TRAILING_PUNCT).strip()
    if key in _YES_TOKENS:
        return "yes"
    if key in _NO_TOKENS:
        return "no"
    return None


def yes_probability_from_logprobs(response: Any) -> Optional[float]:
    """
    P(yes) at the verifier's first verdict token, from its top_logprobs.

    Walks to the first position whose candidate set contains a yes/no
    token -- position 0 for a well-behaved verifier, but a stray leading
    space or newline token should not throw the read away.

    At that position, mass is pooled over all yes-spellings and all
    no-spellings ("Yes", "yes", "YES", "true" are one outcome, not four —
    splitting them would understate whichever the model spread across
    casings) and renormalized over yes+no only. The tail outside the top-k
    is dropped here rather than lumped: it is a constrained one-token
    answer, so mass outside yes/no is off-task noise, not a third verdict.

    Returns None when the response carries no logprobs, or when no
    position offers a yes/no candidate at all.
    """
    for entry in extract_token_logprobs(response):
        pooled = {"yes": 0.0, "no": 0.0}
        found = False

        top = _get(entry, "top_logprobs")
        candidates = top if isinstance(top, (list, tuple)) and top else [entry]

        for cand in candidates:
            key = _verdict_key(_get(cand, "token"))
            if key is None:
                continue
            lp = _get(cand, "logprob")
            if lp is None:
                continue
            pooled[key] += math.exp(float(lp))
            found = True

        if not found:
            continue

        total = pooled["yes"] + pooled["no"]
        if total <= _EPS:
            continue
        return min(1.0, max(0.0, pooled["yes"] / total))

    return None


def parse_verbalized_confidence(text: Any) -> Optional[float]:
    """
    Confidence from the verifier's TEXT, for calls made without logprobs.

    Handles the three shapes a verifier actually emits: a bare verdict
    ("Yes"), a probability ("0.85"), and a percentage ("85%"). A leading
    verdict wins over a trailing number, since "No, about 90% of sources
    disagree" is a NO whose 90% is not a confidence.

    Coarser than the logprob read by construction: a "Yes" collapses to
    1.0 and a "No" to 0.0, which is exactly the overconfidence that makes
    verbalized scoring the fallback rather than the default. Returns None
    when nothing parseable is present.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    stripped = text.strip()
    if not stripped:
        return None

    first_word = stripped.split()[0] if stripped.split() else ""
    verdict = _verdict_key(first_word)
    if verdict is not None:
        return 1.0 if verdict == "yes" else 0.0

    percent = re.search(r"(\d+(?:\.\d+)?)\s*%", stripped)
    if percent:
        return min(1.0, max(0.0, float(percent.group(1)) / 100.0))

    number = re.search(r"[-+]?(?:\d+\.\d+|\.\d+|\d+)", stripped)
    if number:
        value = float(number.group(0))
        if 0.0 <= value <= 1.0:
            return value
        if 0.0 <= value <= 100.0:
            return value / 100.0

    return None


def _verifier_text(response: Any) -> Optional[str]:
    """The verifier's answer text, from any of the shapes we accept."""
    answers = extract_sampled_answers(response)
    for a in answers:
        if isinstance(a, str) and a.strip():
            return a
    return None


@dataclass
class ExternalVerifierResult:
    """Verifier trace for one task.

    ``source`` records WHICH rule produced the number -- "logprob",
    "verbalized", or "binary". Worth keeping per-record: a run where
    half the confidences silently came from the coarse text fallback
    because logprobs were not requested would otherwise look like a
    calibration result rather than a configuration mistake.
    """
    confidence: Optional[float]
    verdict: Optional[str]
    source: Optional[str]
    raw_text: Optional[str] = None
    method: ConfidenceMethod = ConfidenceMethod.EXTERNAL_LLM


def estimate_external_verifier_confidence(
    response: Any,
    *,
    scoring: str = "auto",
    verdict_threshold: float = 0.5,
) -> ExternalVerifierResult:
    """
    Confidence from a cheap verifier model's judgement of the main
    model's answer.

    scoring:
        "auto"       -- P(yes) from logprobs, falling back to the text
                        rules when the call carried no logprobs. The
                        default, and what you want in a run where some
                        calls may be missing logprobs.
        "logprob"    -- P(yes) only; returns None rather than falling
                        back. Use when a run is supposed to have logprobs
                        everywhere and a silent downgrade to the coarse
                        text rule would corrupt the calibration numbers.
        "verbalized" -- text rules only (verdict word, "0.85", "85%").
        "binary"     -- text verdict only, collapsed to 1.0 / 0.0.

    verdict_threshold only labels ``verdict`` for readability; it does
    not touch ``confidence``. The routing threshold is a separate knob
    living on TaskRecord.confidence_threshold.

    Returns confidence=None when the response carries nothing usable --
    missing signal, which compute_ece() drops, not a confident 0.0.
    """
    if scoring not in VERIFIER_SCORINGS:
        raise ValueError(f"scoring must be one of {VERIFIER_SCORINGS}, got {scoring!r}")

    text = _verifier_text(response)
    confidence: Optional[float] = None
    source: Optional[str] = None

    if scoring in ("auto", "logprob"):
        confidence = yes_probability_from_logprobs(response)
        if confidence is not None:
            source = "logprob"

    if confidence is None and scoring in ("auto", "verbalized"):
        confidence = parse_verbalized_confidence(text)
        if confidence is not None:
            source = "verbalized"

    if confidence is None and scoring == "binary":
        verdict_key = _verdict_key(text.split()[0]) if text and text.split() else None
        if verdict_key is not None:
            confidence = 1.0 if verdict_key == "yes" else 0.0
            source = "binary"

    verdict = None
    if confidence is not None:
        verdict = "yes" if confidence >= verdict_threshold else "no"

    return ExternalVerifierResult(
        confidence=confidence,
        verdict=verdict,
        source=source,
        raw_text=text,
    )


def external_verifier_confidence(response: Any, **kwargs: Any) -> Optional[float]:
    """Just the scalar, for callers that only need TaskRecord.confidence_score."""
    return estimate_external_verifier_confidence(response, **kwargs).confidence
