"""
test_confidence.py
==================

Unit tests for eval_harness.confidence, in the same style as
test_metrics.py: fake API responses built by hand, expected confidence
worked out by hand in the comments, asserted against.

No network, no API key, no openai package required -- every fake response
here is a plain dict or SimpleNamespace shaped like what the OpenAI chat
completions API returns with logprobs=True, top_logprobs=k.

Run:
    python -m pytest tests/ -v
    (or, without pytest installed: python -m tests.test_confidence)
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness.confidence import (
    estimate_entropy_confidence,
    token_entropy_confidence,
    normalized_token_entropy,
    extract_token_logprobs,
    estimate_self_consistency_confidence,
    self_consistency_confidence,
    extract_sampled_answers,
    normalize_answer,
)
from eval_harness.models import ConfidenceMethod

LN2 = math.log(2)
LN3 = math.log(3)


# ---------------------------------------------------------------------------
# Fake-response builders
# ---------------------------------------------------------------------------

def token(tok: str, probs, sampled_prob=None):
    """One token position with `probs` as the top_logprobs candidate probs.

    probs are given as PROBABILITIES for readability; they're converted to
    logprobs here, which is what the API actually returns.
    """
    entry = {
        "token": tok,
        "logprob": math.log(sampled_prob if sampled_prob is not None else probs[0]),
        "top_logprobs": [
            {"token": f"{tok}_alt{i}", "logprob": math.log(p)} for i, p in enumerate(probs)
        ],
    }
    return entry


def bare_token(tok: str, sampled_prob: float):
    """A token position with NO top_logprobs -- only the sampled logprob."""
    return {"token": tok, "logprob": math.log(sampled_prob)}


def response(tokens):
    """Wrap token entries in the full choices[0].logprobs.content shape."""
    return {"choices": [{"logprobs": {"content": list(tokens)}}]}


# ---------------------------------------------------------------------------
# Single-position entropy
# ---------------------------------------------------------------------------

def test_entropy_single_certain_token_is_confidence_one():
    # top_logprobs = [p=1.0]. Residual tail = 1 - 1.0 = 0, so the
    # distribution is a single outcome: n_eff = 1 -> entropy 0 by
    # definition -> confidence = 1 - 0 = 1.0
    r = estimate_entropy_confidence(response([token("A", [1.0])]))
    assert r.confidence == 1.0
    assert r.n_tokens_scored == 1
    assert r.method == ConfidenceMethod.ENTROPY
    print("test_entropy_single_certain_token_is_confidence_one: PASS")


def test_entropy_two_equal_candidates_is_confidence_zero():
    # top_logprobs = [0.5, 0.5]. They sum to 1.0 so there is no tail.
    # dist = [0.5, 0.5], n_eff = 2
    # H   = -(0.5*ln0.5 + 0.5*ln0.5) = ln2 = 0.693147
    # max = ln(2)                    = ln2 = 0.693147
    # H_norm = 1.0  ->  confidence = 1 - 1.0 = 0.0
    r = estimate_entropy_confidence(response([token("A", [0.5, 0.5])]))
    assert abs(r.confidence - 0.0) < 1e-12
    print("test_entropy_two_equal_candidates_is_confidence_zero: PASS")


def test_entropy_truncated_tail_is_lumped_by_default():
    # top_logprobs = [0.5, 0.25] -- they sum to 0.75, so 0.25 of the mass
    # fell outside the top-k window. Default (renormalize=False) treats
    # that leftover as one extra outcome:
    #   dist  = [0.5, 0.25, 0.25], n_eff = 3
    #   H     = -(0.5*ln0.5 + 0.25*ln0.25 + 0.25*ln0.25)
    #         = 0.3465735903 + 0.3465735903 + 0.3465735903
    #         = 1.0397207708
    #   max   = ln(3) = 1.0986122887
    #   H_norm = 0.9463938...  ->  confidence = 0.0536061...
    r = estimate_entropy_confidence(response([token("A", [0.5, 0.25])]))

    expected_h = -(0.5 * math.log(0.5) + 0.25 * math.log(0.25) + 0.25 * math.log(0.25))
    expected_conf = 1.0 - expected_h / LN3

    assert abs(r.confidence - expected_conf) < 1e-12
    assert abs(r.confidence - 0.0536061) < 1e-6   # the hand-computed decimal
    print("test_entropy_truncated_tail_is_lumped_by_default: PASS")


def test_entropy_renormalize_discards_tail():
    # Same [0.5, 0.25] input, but renormalize=True rescales to sum 1:
    #   dist  = [2/3, 1/3], n_eff = 2
    #   H     = -(2/3*ln(2/3) + 1/3*ln(1/3)) = 0.2703100721 + 0.3662040962
    #         = 0.6365141683
    #   max   = ln(2) = 0.6931471806
    #   H_norm = 0.9182958341  ->  confidence = 0.0817041659
    # Note this reads as MORE confident than the lumped version above
    # (0.0817 > 0.0536): discarding the tail hides real uncertainty,
    # which is exactly why lumping is the default.
    r = estimate_entropy_confidence(response([token("A", [0.5, 0.25])]), renormalize=True)

    expected_h = -((2 / 3) * math.log(2 / 3) + (1 / 3) * math.log(1 / 3))
    expected_conf = 1.0 - expected_h / LN2

    assert abs(r.confidence - expected_conf) < 1e-12
    assert abs(r.confidence - 0.0817041659) < 1e-9
    assert r.confidence > 0.0536061  # tail-discarding is the optimistic reading
    print("test_entropy_renormalize_discards_tail: PASS")


def test_entropy_falls_back_to_sampled_token_logprob():
    # No top_logprobs at all -- only the sampled token's logprob, p = 0.8.
    # Falls back to a two-outcome view: [chosen 0.8, everything-else 0.2]
    #   H     = -(0.8*ln0.8 + 0.2*ln0.2) = 0.178514841 + 0.3218875825
    #         = 0.5004024235
    #   max   = ln(2) = 0.6931471806
    #   H_norm = 0.7219280949  ->  confidence = 0.2780719051
    r = estimate_entropy_confidence(response([bare_token("A", 0.8)]))

    expected_h = -(0.8 * math.log(0.8) + 0.2 * math.log(0.2))
    expected_conf = 1.0 - expected_h / LN2

    assert abs(r.confidence - expected_conf) < 1e-12
    assert abs(r.confidence - 0.2780719051) < 1e-9
    print("test_entropy_falls_back_to_sampled_token_logprob: PASS")


def test_normalization_is_independent_of_top_k():
    # A flat distribution should score 0.0 confidence regardless of how
    # many alternatives the API was asked for, because we divide by
    # log(n_eff) rather than by a fixed constant. Uniform over 2 and
    # uniform over 4 both give H_norm = 1.0.
    assert abs(normalized_token_entropy([0.5, 0.5]) - 1.0) < 1e-12
    assert abs(normalized_token_entropy([0.25] * 4) - 1.0) < 1e-12
    assert abs(normalized_token_entropy([0.1] * 10) - 1.0) < 1e-12
    print("test_normalization_is_independent_of_top_k: PASS")


# ---------------------------------------------------------------------------
# Aggregation across tokens
# ---------------------------------------------------------------------------

def _two_token_response():
    # token 1: certain          -> H_norm = 0.0
    # token 2: two equal cands  -> H_norm = 1.0
    return response([token("A", [1.0]), token("B", [0.5, 0.5])])


def test_mean_aggregation():
    # mean H_norm = (0.0 + 1.0) / 2 = 0.5  ->  confidence = 0.5
    r = estimate_entropy_confidence(_two_token_response(), aggregation="mean")
    assert abs(r.confidence - 0.5) < 1e-12
    assert abs(r.mean_normalized_entropy - 0.5) < 1e-12
    assert r.n_tokens_scored == 2
    print("test_mean_aggregation: PASS")


def test_min_aggregation_takes_worst_token():
    # worst token has H_norm = 1.0  ->  confidence = 1 - 1.0 = 0.0
    # (strictly below the mean-aggregated 0.5: one bad token sinks it)
    r = estimate_entropy_confidence(_two_token_response(), aggregation="min")
    assert abs(r.confidence - 0.0) < 1e-12
    # mean is still reported alongside, unchanged by the aggregation choice
    assert abs(r.mean_normalized_entropy - 0.5) < 1e-12
    print("test_min_aggregation_takes_worst_token: PASS")


def test_first_aggregation_stops_at_first_token():
    # first token has H_norm = 0.0  ->  confidence = 1.0, and the second
    # token is never scored
    r = estimate_entropy_confidence(_two_token_response(), aggregation="first")
    assert abs(r.confidence - 1.0) < 1e-12
    assert r.n_tokens_scored == 1
    print("test_first_aggregation_stops_at_first_token: PASS")


def test_skip_whitespace_excludes_blank_tokens():
    # " " is a certain whitespace token (H_norm = 0.0). Including it drags
    # the mean toward confident:  (0.0 + 1.0) / 2 = 0.5  -> conf 0.5
    # Skipping it leaves only the uncertain token: H_norm = 1.0 -> conf 0.0
    resp = response([token(" ", [1.0]), token("B", [0.5, 0.5])])

    keep = estimate_entropy_confidence(resp, skip_whitespace=False)
    drop = estimate_entropy_confidence(resp, skip_whitespace=True)

    assert abs(keep.confidence - 0.5) < 1e-12
    assert abs(drop.confidence - 0.0) < 1e-12
    assert drop.n_tokens_scored == 1
    assert drop.n_tokens_skipped == 1
    print("test_skip_whitespace_excludes_blank_tokens: PASS")


# ---------------------------------------------------------------------------
# Input shapes and degenerate input
# ---------------------------------------------------------------------------

def test_accepts_dict_object_and_bare_list_shapes():
    # The SDK returns pydantic objects; cached JSON gives dicts. Both must
    # produce the identical score, as must the bare token list.
    entries = [token("A", [0.5, 0.25])]

    as_dict = response(entries)
    as_object = SimpleNamespace(
        choices=[SimpleNamespace(logprobs=SimpleNamespace(content=[
            SimpleNamespace(
                token="A",
                logprob=math.log(0.5),
                top_logprobs=[
                    SimpleNamespace(token="A_alt0", logprob=math.log(0.5)),
                    SimpleNamespace(token="A_alt1", logprob=math.log(0.25)),
                ],
            )
        ]))]
    )

    c_dict = token_entropy_confidence(as_dict)
    c_obj = token_entropy_confidence(as_object)
    c_list = token_entropy_confidence(entries)

    assert abs(c_dict - c_obj) < 1e-12
    assert abs(c_dict - c_list) < 1e-12
    assert len(extract_token_logprobs(as_dict)) == 1
    print("test_accepts_dict_object_and_bare_list_shapes: PASS")


def test_missing_logprobs_returns_none_not_zero():
    # A response created WITHOUT logprobs=True carries no signal. That is
    # not the same as "confidently wrong", so we return None -- which
    # compute_ece() drops rather than binning at 0.0.
    assert token_entropy_confidence({"choices": [{"message": {"content": "hi"}}]}) is None
    assert token_entropy_confidence(response([])) is None
    assert token_entropy_confidence(None) is None
    assert normalized_token_entropy([]) is None
    print("test_missing_logprobs_returns_none_not_zero: PASS")


def test_rejects_unknown_aggregation():
    try:
        estimate_entropy_confidence(response([token("A", [1.0])]), aggregation="median")
    except ValueError:
        print("test_rejects_unknown_aggregation: PASS")
        return
    raise AssertionError("expected ValueError for unknown aggregation")


# ---------------------------------------------------------------------------
# (b) Self-consistency -- fake-response builders
# ---------------------------------------------------------------------------

def sampled(*answers):
    """One response sampled with n=k: choices[*].message.content."""
    return {"choices": [{"message": {"content": a}} for a in answers]}


def separate_calls(*answers):
    """k separate single-choice responses, the other shape callers use."""
    return [{"choices": [{"message": {"content": a}}]} for a in answers]


# ---------------------------------------------------------------------------
# Self-consistency: agreement scoring
# ---------------------------------------------------------------------------

def test_unanimous_samples_are_fully_confident():
    # 5/5 agree -> agreement 1.0
    r = estimate_self_consistency_confidence(["42", "42", "42", "42", "42"])
    assert r.confidence == 1.0
    assert r.consensus_answer == "42"
    assert r.n_samples == 5
    assert r.method is ConfidenceMethod.SELF_CONSISTENCY
    print("test_unanimous_samples_are_fully_confident: PASS")


def test_majority_agreement_rate():
    # 3 of 5 agree on "42" -> 0.6, and the consensus is the majority answer
    r = estimate_self_consistency_confidence(["42", "42", "42", "43", "44"])
    assert abs(r.confidence - 0.6) < 1e-9
    assert r.consensus_answer == "42"
    assert r.vote_counts[0] == ("42", 3)
    print("test_majority_agreement_rate: PASS")


def test_total_disagreement_is_floor():
    # k distinct answers -> agreement 1/k, the floor, never 0
    r = estimate_self_consistency_confidence(["a", "b", "c", "d"])
    assert abs(r.confidence - 0.25) < 1e-9
    print("test_total_disagreement_is_floor: PASS")


def test_ties_break_by_first_appearance():
    # 2-2 tie: "b" appears first, so it wins deterministically regardless
    # of dict/Counter ordering
    r = estimate_self_consistency_confidence(["b", "a", "b", "a"])
    assert r.consensus_answer == "b"
    assert abs(r.agreement_rate - 0.5) < 1e-9
    # same votes, reversed order -> the other answer wins, still deterministic
    r2 = estimate_self_consistency_confidence(["a", "b", "a", "b"])
    assert r2.consensus_answer == "a"
    print("test_ties_break_by_first_appearance: PASS")


# ---------------------------------------------------------------------------
# Self-consistency: answer normalization
# ---------------------------------------------------------------------------

def test_normalization_buckets_equivalent_spellings():
    # all five of these are the same answer; an over-strict comparison
    # would report 1/5 agreement and make the model look uncalibrated
    r = estimate_self_consistency_confidence(
        ["42", " 42 ", "42.", "The answer is 42", "Final answer: 42"]
    )
    assert r.confidence == 1.0, r.vote_counts
    assert r.consensus_answer == "42"
    print("test_normalization_buckets_equivalent_spellings: PASS")


def test_numeric_spellings_are_canonicalized():
    # "$1,234.00" == "1234" == "1234.0"
    r = estimate_self_consistency_confidence(["$1,234.00", "1234", "1234.0"])
    assert r.confidence == 1.0, r.vote_counts
    assert r.consensus_answer == "1234"
    print("test_numeric_spellings_are_canonicalized: PASS")


def test_normalization_is_case_insensitive():
    r = estimate_self_consistency_confidence(["Paris", "paris", "PARIS"])
    assert r.confidence == 1.0
    assert r.consensus_answer == "paris"
    print("test_normalization_is_case_insensitive: PASS")


def test_distinct_answers_stay_distinct():
    # normalization must not over-merge: 42 and 43 are different answers
    r = estimate_self_consistency_confidence(["42", "43"])
    assert abs(r.confidence - 0.5) < 1e-9
    assert len(r.vote_counts) == 2
    print("test_distinct_answers_stay_distinct: PASS")


def test_custom_normalizer_is_honored():
    # a task-specific grader: bucket by first character only
    r = estimate_self_consistency_confidence(
        ["apple", "avocado", "banana"], normalizer=lambda s: s[0] if s else None
    )
    assert abs(r.confidence - (2 / 3)) < 1e-9
    assert r.consensus_answer == "a"
    print("test_custom_normalizer_is_honored: PASS")


# ---------------------------------------------------------------------------
# Self-consistency: missing / unusable samples
# ---------------------------------------------------------------------------

def test_blank_samples_are_skipped_not_voted():
    # None and "" must not become a vote for the empty answer: 2 valid
    # samples both saying 42 -> 1.0, with 2 skipped recorded
    r = estimate_self_consistency_confidence(["42", None, "42", "   "])
    assert r.confidence == 1.0
    assert r.n_samples == 2
    assert r.n_skipped == 2
    print("test_blank_samples_are_skipped_not_voted: PASS")


def test_no_usable_samples_returns_none():
    # missing signal, not a confident-wrong 0.0 -- compute_ece drops None
    r = estimate_self_consistency_confidence([None, "", "  "])
    assert r.confidence is None
    assert r.consensus_answer is None
    assert r.agreement_rate is None
    assert r.n_skipped == 3
    print("test_no_usable_samples_returns_none: PASS")


def test_empty_input_returns_none():
    assert estimate_self_consistency_confidence([]).confidence is None
    assert estimate_self_consistency_confidence(None).confidence is None
    print("test_empty_input_returns_none: PASS")


# ---------------------------------------------------------------------------
# Self-consistency: response shapes
# ---------------------------------------------------------------------------

def test_reads_n_equals_k_response_shape():
    # one request sampled n=4
    r = estimate_self_consistency_confidence(sampled("42", "42", "42", "7"))
    assert abs(r.confidence - 0.75) < 1e-9
    assert r.consensus_answer == "42"
    print("test_reads_n_equals_k_response_shape: PASS")


def test_reads_separate_call_response_shape():
    # k separate single-choice responses
    r = estimate_self_consistency_confidence(separate_calls("42", "42", "7", "7"))
    assert r.n_samples == 4
    assert abs(r.agreement_rate - 0.5) < 1e-9
    print("test_reads_separate_call_response_shape: PASS")


def test_reads_sdk_object_shape():
    # the SDK returns pydantic objects, not dicts -- same attribute path
    resp = SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content="42")),
        SimpleNamespace(message=SimpleNamespace(content="42")),
    ])
    r = estimate_self_consistency_confidence(resp)
    assert r.confidence == 1.0
    assert r.n_samples == 2
    print("test_reads_sdk_object_shape: PASS")


def test_extract_sampled_answers_shapes():
    assert extract_sampled_answers(["a", "b"]) == ["a", "b"]
    assert extract_sampled_answers("a") == ["a"]
    assert extract_sampled_answers(sampled("a", "b")) == ["a", "b"]
    assert extract_sampled_answers(separate_calls("a", "b")) == ["a", "b"]
    assert extract_sampled_answers(None) == []
    print("test_extract_sampled_answers_shapes: PASS")


# ---------------------------------------------------------------------------
# Self-consistency: entropy scoring
# ---------------------------------------------------------------------------

def test_entropy_scoring_penalizes_scatter_below_clean_tie():
    # this is the whole reason entropy scoring exists, and the reason it
    # normalizes by log(k) rather than log(distinct answers).
    # 10 samples, 5-5 against one rival:
    #   H = ln2 = 0.6931, /ln10 = 0.3010 -> confidence 0.6990
    # 10 samples, 5 vs five singletons:
    #   H = 0.5*ln2 + 5*(0.1*ln10) = 1.4979, /ln10 = 0.6505 -> 0.3495
    # both are 0.5 agreement; the scatter must score strictly lower.
    tie = ["a"] * 5 + ["b"] * 5
    scatter = ["a"] * 5 + ["b", "c", "d", "e", "f"]

    r_tie = estimate_self_consistency_confidence(tie, scoring="entropy")
    r_scatter = estimate_self_consistency_confidence(scatter, scoring="entropy")

    assert abs(r_tie.confidence - 0.6990) < 1e-3, r_tie.confidence
    assert abs(r_scatter.confidence - 0.3495) < 1e-3, r_scatter.confidence
    assert r_scatter.confidence < r_tie.confidence

    # agreement scoring cannot tell them apart -- that is the gap being filled
    assert abs(r_tie.agreement_rate - r_scatter.agreement_rate) < 1e-9
    print("test_entropy_scoring_penalizes_scatter_below_clean_tie: PASS")


def test_entropy_scoring_unanimous_is_one():
    r = estimate_self_consistency_confidence(["42"] * 8, scoring="entropy")
    assert abs(r.confidence - 1.0) < 1e-9
    print("test_entropy_scoring_unanimous_is_one: PASS")


def test_entropy_scoring_all_distinct_is_zero():
    # k samples, k distinct answers -> H = ln(k) -> normalized 1.0 -> conf 0
    r = estimate_self_consistency_confidence(["a", "b", "c", "d"], scoring="entropy")
    assert abs(r.confidence - 0.0) < 1e-9
    print("test_entropy_scoring_all_distinct_is_zero: PASS")


def test_entropy_and_agreement_agree_at_the_extremes():
    # the two rules only diverge in the middle; unanimity is 1.0 for both
    unanimous = ["x"] * 6
    assert estimate_self_consistency_confidence(unanimous).confidence == 1.0
    assert abs(
        estimate_self_consistency_confidence(unanimous, scoring="entropy").confidence - 1.0
    ) < 1e-9
    print("test_entropy_and_agreement_agree_at_the_extremes: PASS")


def test_single_sample_scores_one_under_both_rules():
    # documented caveat: k=1 always agrees with itself and carries no
    # self-consistency signal -- must not crash on log(1) either
    assert estimate_self_consistency_confidence(["42"]).confidence == 1.0
    assert estimate_self_consistency_confidence(["42"], scoring="entropy").confidence == 1.0
    print("test_single_sample_scores_one_under_both_rules: PASS")


# ---------------------------------------------------------------------------
# Self-consistency: API surface
# ---------------------------------------------------------------------------

def test_rejects_unknown_scoring():
    try:
        estimate_self_consistency_confidence(["a"], scoring="majority")
    except ValueError:
        print("test_rejects_unknown_scoring: PASS")
        return
    raise AssertionError("expected ValueError for unknown scoring")


def test_scalar_wrapper_matches_full_result():
    samples = ["42", "42", "7"]
    assert self_consistency_confidence(samples) == (
        estimate_self_consistency_confidence(samples).confidence
    )
    print("test_scalar_wrapper_matches_full_result: PASS")


def test_normalize_answer_directly():
    assert normalize_answer("  The answer is 42.  ") == "42"
    assert normalize_answer("1,000") == "1000"
    assert normalize_answer("3.50") == "3.5"
    assert normalize_answer("f(x)") == "f(x)"     # closing paren is content
    assert normalize_answer(None) is None
    assert normalize_answer("   ") is None
    assert normalize_answer(42) == "42"           # non-str coerced
    print("test_normalize_answer_directly: PASS")


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
