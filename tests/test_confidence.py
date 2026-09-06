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
    estimate_hybrid_confidence,
    hybrid_confidence,
    HYBRID_COMBINERS,
    estimate_entropy_confidence,
    token_entropy_confidence,
    normalized_token_entropy,
    extract_token_logprobs,
    estimate_self_consistency_confidence,
    self_consistency_confidence,
    extract_sampled_answers,
    normalize_answer,
    estimate_external_verifier_confidence,
    external_verifier_confidence,
    build_verifier_messages,
    parse_verbalized_confidence,
    yes_probability_from_logprobs,
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


# ---------------------------------------------------------------------------
# (c) External verifier -- fake-response builders
# ---------------------------------------------------------------------------

def verdict_response(text, probs=None):
    """A verifier response: `text` as the message, optional top_logprobs
    on the first token. `probs` maps token -> probability."""
    resp = {"choices": [{"message": {"content": text}}]}
    if probs is not None:
        resp["choices"][0]["logprobs"] = {"content": [{
            "token": text.split()[0] if text.split() else text,
            "logprob": math.log(max(list(probs.values()) + [1e-12])),
            "top_logprobs": [
                {"token": t, "logprob": math.log(p) if p > 0 else -100.0}
                for t, p in probs.items()
            ],
        }]}
    return resp


# ---------------------------------------------------------------------------
# External verifier: prompt construction
# ---------------------------------------------------------------------------

def test_verifier_messages_shape():
    msgs = build_verifier_messages("What is 2+2?", "4")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "2+2" in msgs[1]["content"]
    assert "4" in msgs[1]["content"]
    # the Yes/No constraint is what makes the logprob read possible
    assert "Yes" in msgs[0]["content"] and "No" in msgs[0]["content"]
    print("test_verifier_messages_shape: PASS")


def test_verifier_messages_handle_missing_answer():
    msgs = build_verifier_messages("What is 2+2?", None)
    assert "(no answer given)" in msgs[1]["content"]
    print("test_verifier_messages_handle_missing_answer: PASS")


def test_gold_answer_is_labelled_as_oracle():
    # leaking the label into a live run invalidates the experiment, so it
    # has to be conspicuous in a logged trace, not silent
    msgs = build_verifier_messages("q", "a", gold_answer="42")
    assert "ORACLE" in msgs[1]["content"]
    assert "42" in msgs[1]["content"]
    # and absent entirely when not asked for
    assert "ORACLE" not in build_verifier_messages("q", "a")[1]["content"]
    print("test_gold_answer_is_labelled_as_oracle: PASS")


# ---------------------------------------------------------------------------
# External verifier: logprob scoring
# ---------------------------------------------------------------------------

def test_yes_probability_from_logprobs():
    # P(Yes)=0.8, P(No)=0.2 -> renormalized over yes+no -> 0.8
    r = estimate_external_verifier_confidence(
        verdict_response("Yes", {"Yes": 0.8, "No": 0.2})
    )
    assert abs(r.confidence - 0.8) < 1e-9
    assert r.source == "logprob"
    assert r.verdict == "yes"
    assert r.method is ConfidenceMethod.EXTERNAL_LLM
    print("test_yes_probability_from_logprobs: PASS")


def test_logprob_renormalizes_over_yes_no_only():
    # off-task mass ("Maybe") is dropped, not treated as a third verdict:
    # 0.6 / (0.6 + 0.2) = 0.75
    r = estimate_external_verifier_confidence(
        verdict_response("Yes", {"Yes": 0.6, "No": 0.2, "Maybe": 0.2})
    )
    assert abs(r.confidence - 0.75) < 1e-9
    print("test_logprob_renormalizes_over_yes_no_only: PASS")


def test_yes_spellings_are_pooled():
    # "Yes"/"yes"/"YES" are one outcome, not three -- splitting them would
    # understate a model that spread mass across casings:
    # (0.3+0.2+0.1) / (0.6 + 0.4) = 0.6
    r = estimate_external_verifier_confidence(
        verdict_response("Yes", {"Yes": 0.3, "yes": 0.2, "YES": 0.1, "No": 0.4})
    )
    assert abs(r.confidence - 0.6) < 1e-9
    print("test_yes_spellings_are_pooled: PASS")


def test_low_yes_probability_reads_as_no():
    r = estimate_external_verifier_confidence(
        verdict_response("No", {"Yes": 0.1, "No": 0.9})
    )
    assert abs(r.confidence - 0.1) < 1e-9
    assert r.verdict == "no"
    print("test_low_yes_probability_reads_as_no: PASS")


def test_logprob_read_skips_leading_non_verdict_token():
    # a stray newline token before the verdict must not throw the read away
    resp = {"choices": [{
        "message": {"content": "Yes"},
        "logprobs": {"content": [
            {"token": "\n", "logprob": math.log(0.99),
             "top_logprobs": [{"token": "\n", "logprob": math.log(0.99)}]},
            {"token": "Yes", "logprob": math.log(0.7),
             "top_logprobs": [{"token": "Yes", "logprob": math.log(0.7)},
                              {"token": "No", "logprob": math.log(0.3)}]},
        ]},
    }]}
    r = estimate_external_verifier_confidence(resp)
    assert abs(r.confidence - 0.7) < 1e-9
    assert r.source == "logprob"
    print("test_logprob_read_skips_leading_non_verdict_token: PASS")


def test_logprob_scoring_does_not_fall_back():
    # scoring="logprob" must return None rather than silently downgrading
    # to the coarse text rule -- a silent downgrade corrupts calibration
    r = estimate_external_verifier_confidence(
        verdict_response("Yes"), scoring="logprob"
    )
    assert r.confidence is None
    assert r.source is None
    # ...whereas "auto" does fall back, and records that it did
    r_auto = estimate_external_verifier_confidence(verdict_response("Yes"))
    assert r_auto.confidence == 1.0
    assert r_auto.source == "verbalized"
    print("test_logprob_scoring_does_not_fall_back: PASS")


def test_logprob_preferred_over_text_when_both_present():
    # text says a flat "Yes" (would be 1.0); logprobs say 0.65 -- the
    # calibrated number must win
    r = estimate_external_verifier_confidence(
        verdict_response("Yes", {"Yes": 0.65, "No": 0.35})
    )
    assert abs(r.confidence - 0.65) < 1e-9
    assert r.source == "logprob"
    print("test_logprob_preferred_over_text_when_both_present: PASS")


# ---------------------------------------------------------------------------
# External verifier: verbalized scoring
# ---------------------------------------------------------------------------

def test_verbalized_verdict_words():
    assert parse_verbalized_confidence("Yes") == 1.0
    assert parse_verbalized_confidence("no") == 0.0
    assert parse_verbalized_confidence("  Correct.  ") == 1.0
    assert parse_verbalized_confidence("Incorrect") == 0.0
    print("test_verbalized_verdict_words: PASS")


def test_verbalized_numeric_scores():
    assert abs(parse_verbalized_confidence("0.85") - 0.85) < 1e-9
    assert abs(parse_verbalized_confidence("85%") - 0.85) < 1e-9
    assert abs(parse_verbalized_confidence("Confidence: 72%") - 0.72) < 1e-9
    print("test_verbalized_numeric_scores: PASS")


def test_leading_verdict_beats_trailing_number():
    # "No, about 90% of sources disagree" is a NO; the 90% is not a
    # confidence and must not be read as one
    assert parse_verbalized_confidence("No, about 90% of sources disagree") == 0.0
    print("test_leading_verdict_beats_trailing_number: PASS")


def test_verbalized_unparseable_returns_none():
    assert parse_verbalized_confidence("I cannot determine this") is None
    assert parse_verbalized_confidence("") is None
    assert parse_verbalized_confidence(None) is None
    print("test_verbalized_unparseable_returns_none: PASS")


def test_binary_scoring_collapses_to_extremes():
    r = estimate_external_verifier_confidence(verdict_response("Yes"), scoring="binary")
    assert r.confidence == 1.0 and r.source == "binary"
    r2 = estimate_external_verifier_confidence(verdict_response("No"), scoring="binary")
    assert r2.confidence == 0.0
    print("test_binary_scoring_collapses_to_extremes: PASS")


# ---------------------------------------------------------------------------
# External verifier: missing signal + API surface
# ---------------------------------------------------------------------------

def test_unusable_verifier_response_returns_none():
    # missing signal, not a confident 0.0
    r = estimate_external_verifier_confidence(verdict_response("I don't know"))
    assert r.confidence is None
    assert r.verdict is None
    assert r.source is None
    assert estimate_external_verifier_confidence(None).confidence is None
    print("test_unusable_verifier_response_returns_none: PASS")


def test_source_records_which_rule_fired():
    # a run that silently used the coarse fallback everywhere should be
    # diagnosable from the records, not look like a calibration result
    with_lp = estimate_external_verifier_confidence(
        verdict_response("Yes", {"Yes": 0.9, "No": 0.1})
    )
    without_lp = estimate_external_verifier_confidence(verdict_response("Yes"))
    assert with_lp.source == "logprob"
    assert without_lp.source == "verbalized"
    print("test_source_records_which_rule_fired: PASS")


def test_verifier_reads_sdk_object_shape():
    resp = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="Yes"),
        logprobs=SimpleNamespace(content=[SimpleNamespace(
            token="Yes", logprob=math.log(0.75),
            top_logprobs=[SimpleNamespace(token="Yes", logprob=math.log(0.75)),
                          SimpleNamespace(token="No", logprob=math.log(0.25))],
        )]),
    )])
    r = estimate_external_verifier_confidence(resp)
    assert abs(r.confidence - 0.75) < 1e-9
    print("test_verifier_reads_sdk_object_shape: PASS")


def test_verdict_threshold_labels_only():
    # threshold moves the label, never the number
    resp = verdict_response("Yes", {"Yes": 0.6, "No": 0.4})
    low = estimate_external_verifier_confidence(resp, verdict_threshold=0.5)
    high = estimate_external_verifier_confidence(resp, verdict_threshold=0.9)
    assert low.verdict == "yes" and high.verdict == "no"
    assert abs(low.confidence - high.confidence) < 1e-12
    print("test_verdict_threshold_labels_only: PASS")


def test_verifier_rejects_unknown_scoring():
    try:
        estimate_external_verifier_confidence(verdict_response("Yes"), scoring="vibes")
    except ValueError:
        print("test_verifier_rejects_unknown_scoring: PASS")
        return
    raise AssertionError("expected ValueError for unknown scoring")


def test_verifier_scalar_wrapper_matches_full_result():
    resp = verdict_response("Yes", {"Yes": 0.7, "No": 0.3})
    assert external_verifier_confidence(resp) == (
        estimate_external_verifier_confidence(resp).confidence
    )
    print("test_verifier_scalar_wrapper_matches_full_result: PASS")


# ---------------------------------------------------------------------------
# (d) Hybrid combiner
# ---------------------------------------------------------------------------

def test_hybrid_equal_weights_is_plain_mean():
    # (0.9 + 0.6 + 0.3) / 3 = 0.6
    r = estimate_hybrid_confidence({"a": 0.9, "b": 0.6, "c": 0.3})
    assert abs(r.confidence - 0.6) < 1e-12
    assert r.method == ConfidenceMethod.HYBRID
    assert r.missing == []
    print("test_hybrid_equal_weights_is_plain_mean: PASS")


def test_hybrid_weighted_mean():
    # (0.8*3 + 0.4*1) / 4 = 2.8 / 4 = 0.7
    r = estimate_hybrid_confidence(
        {"entropy": 0.8, "verifier": 0.4},
        weights={"entropy": 3.0, "verifier": 1.0},
    )
    assert abs(r.confidence - 0.7) < 1e-12
    # Reported weights are renormalized to sum to 1.
    assert abs(r.weights["entropy"] - 0.75) < 1e-12
    assert abs(sum(r.weights.values()) - 1.0) < 1e-12
    print("test_hybrid_weighted_mean: PASS")


def test_hybrid_missing_signal_renormalizes_rather_than_counting_as_zero():
    # The verifier is absent, not negative. Mean over the two present
    # signals is 0.7 -- NOT (0.8 + 0.6 + 0) / 3 = 0.466...
    r = estimate_hybrid_confidence({"a": 0.8, "b": 0.6, "verifier": None})
    assert abs(r.confidence - 0.7) < 1e-12
    assert r.missing == ["verifier"]
    assert set(r.used) == {"a", "b"}
    assert abs(sum(r.weights.values()) - 1.0) < 1e-12
    print("test_hybrid_missing_signal_renormalizes_rather_than_counting_as_zero: PASS")


def test_hybrid_missing_signal_drops_its_weight_too():
    # Weighted mean with the heavy signal absent must fall back to the
    # light one alone, not scale it by its original small weight.
    r = estimate_hybrid_confidence(
        {"heavy": None, "light": 0.5},
        weights={"heavy": 9.0, "light": 1.0},
    )
    assert abs(r.confidence - 0.5) < 1e-12
    assert r.weights == {"light": 1.0}
    print("test_hybrid_missing_signal_drops_its_weight_too: PASS")


def test_hybrid_all_missing_returns_none_not_zero():
    r = estimate_hybrid_confidence({"a": None, "b": None})
    assert r.confidence is None
    assert r.used == {}
    assert sorted(r.missing) == ["a", "b"]
    print("test_hybrid_all_missing_returns_none_not_zero: PASS")


def test_hybrid_empty_signals_returns_none():
    assert estimate_hybrid_confidence({}).confidence is None
    print("test_hybrid_empty_signals_returns_none: PASS")


def test_hybrid_min_takes_the_least_confident_signal():
    r = estimate_hybrid_confidence({"a": 0.9, "b": 0.2, "c": 0.7}, combine="min")
    assert r.confidence == 0.2
    print("test_hybrid_min_takes_the_least_confident_signal: PASS")


def test_hybrid_min_ignores_weights():
    # Weighting the confident signal heavily must not rescue the score:
    # "is ANY estimator unsure" is the whole point of the min rule.
    r = estimate_hybrid_confidence(
        {"a": 0.9, "b": 0.2},
        weights={"a": 100.0, "b": 1.0},
        combine="min",
    )
    assert r.confidence == 0.2
    print("test_hybrid_min_ignores_weights: PASS")


def test_hybrid_min_skips_missing_signals():
    r = estimate_hybrid_confidence({"a": 0.4, "b": None}, combine="min")
    assert r.confidence == 0.4
    assert r.missing == ["b"]
    print("test_hybrid_min_skips_missing_signals: PASS")


def test_hybrid_accepts_estimator_result_objects_directly():
    # The three estimators' results are passed straight through; the
    # combiner reads .confidence off each rather than needing scalars.
    entropy = estimate_entropy_confidence(
        response([token("A", [0.7, 0.3])])
    )
    verifier = estimate_external_verifier_confidence(
        verdict_response("Yes", {"Yes": 0.75, "No": 0.25})
    )
    r = estimate_hybrid_confidence(
        {ConfidenceMethod.ENTROPY: entropy, ConfidenceMethod.EXTERNAL_LLM: verifier}
    )
    # Keys come out as the enum's string value, not repr(enum).
    assert set(r.used) == {"entropy", "external_llm"}
    expected = (entropy.confidence + verifier.confidence) / 2
    assert abs(r.confidence - expected) < 1e-12
    print("test_hybrid_accepts_estimator_result_objects_directly: PASS")


def test_hybrid_clamps_out_of_range_signals():
    r = estimate_hybrid_confidence({"a": 1.4, "b": -0.2})
    # Clamped to 1.0 and 0.0 before averaging -> 0.5, and never outside [0, 1].
    assert abs(r.confidence - 0.5) < 1e-12
    assert 0.0 <= r.confidence <= 1.0
    print("test_hybrid_clamps_out_of_range_signals: PASS")


def test_hybrid_rejects_unknown_combine():
    try:
        estimate_hybrid_confidence({"a": 0.5}, combine="median")
    except ValueError:
        print("test_hybrid_rejects_unknown_combine: PASS")
        return
    raise AssertionError("expected ValueError for unknown combine")


def test_hybrid_rejects_negative_weight():
    try:
        estimate_hybrid_confidence({"a": 0.5}, weights={"a": -1.0})
    except ValueError:
        print("test_hybrid_rejects_negative_weight: PASS")
        return
    raise AssertionError("expected ValueError for negative weight")


def test_hybrid_rejects_all_zero_weights():
    # Would be a divide-by-zero rather than a meaningful combination.
    try:
        estimate_hybrid_confidence({"a": 0.5, "b": 0.5}, weights={"a": 0.0, "b": 0.0})
    except ValueError:
        print("test_hybrid_rejects_all_zero_weights: PASS")
        return
    raise AssertionError("expected ValueError for zero total weight")


def test_hybrid_rejects_non_numeric_signal():
    try:
        estimate_hybrid_confidence({"a": "0.5"})
    except TypeError:
        print("test_hybrid_rejects_non_numeric_signal: PASS")
        return
    raise AssertionError("expected TypeError for a string signal")


def test_hybrid_rejects_nan_signal():
    # NaN would propagate silently through the mean and poison ECE.
    try:
        estimate_hybrid_confidence({"a": float("nan")})
    except ValueError:
        print("test_hybrid_rejects_nan_signal: PASS")
        return
    raise AssertionError("expected ValueError for NaN signal")


def test_hybrid_combiners_tuple_matches_accepted_values():
    for rule in HYBRID_COMBINERS:
        estimate_hybrid_confidence({"a": 0.5}, combine=rule)
    print("test_hybrid_combiners_tuple_matches_accepted_values: PASS")


def test_hybrid_scalar_wrapper_matches_full_result():
    signals = {"a": 0.9, "b": 0.3}
    assert hybrid_confidence(signals) == estimate_hybrid_confidence(signals).confidence
    print("test_hybrid_scalar_wrapper_matches_full_result: PASS")


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
