"""
E4 - Zero-Hallucination Guardrail Execution.

Implements the exact algorithm from the methodology (Section 4, Phase 3):

    Input: User Query (Q), Retrieved Context Chunks (C)
    Output: Verified Response (R_verified) or Refinement Trigger

    1. Generate Candidate Response: R_candidate = LLM_Agent(Q, C, CoT_Prompt)
    2. Extract Triples from Context: Graph_C = Extract_Triples(C)
    3. Extract Triples from Response: Graph_R = Extract_Triples(R_candidate)
    4. Perform Alignment Check: Mismatch_Score = Compute_Structural_Distance(Graph_R, Graph_C)
    5. If Mismatch_Score == 0:
           Return R_candidate As R_verified
       Else:
           Initiate Query_Refinement_Loop(Q, Feedback=Mismatch_Details)

This wraps Phase 2's TheologicalAgent directly: R_candidate comes from its
CoT reasoning step, Q/C come from its retrieval loop. Refinement re-uses
Phase 2's own D1 agent loop + C3 query refinement rather than duplicating
that logic - Phase 3 adds the verification layer on top, it doesn't
reimplement retrieval.

Requirements:
    pip install networkx rapidfuzz pyarabic
"""

import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

from e2_knowledge_graph import build_context_graph, build_response_graph
from e3_structural_alignment import compute_structural_alignment, AlignmentResult, print_alignment_result


@dataclass
class GuardrailConfig:
    similarity_threshold: float = 65.0   # per-triple alignment threshold (E3)
    max_refinement_attempts: int = 2     # hard cap - mirrors D1's bounded loop philosophy
    max_acceptable_mismatch: float = 0.0  # 0.0 = strict (methodology default); raise to allow
                                           # partially-grounded answers through, e.g. 0.2 accepts
                                           # a response where up to 20% of claims are unaligned
    verbose: bool = True


@dataclass
class GuardrailResult:
    query: str
    verified: bool
    final_response: Optional[str]
    final_mismatch_score: Optional[float]
    attempts: List[Dict] = field(default_factory=list)  # per-attempt log for auditability
    stopped_reason: str = ""  # "verified" | "max_attempts_reached" | "insufficient_info"
    partially_verified: bool = False  # True if accepted under a nonzero max_acceptable_mismatch
    unverified_claims: List = field(default_factory=list)  # the specific mismatched triples, for transparency


INSUFFICIENT_INFO_MARKER = "غير كاف"


def is_insufficient_info_response(response_text: str) -> bool:
    """
    Detects the model honestly saying "I can't answer from this context"
    (per D2's prompt instructions), rather than fabricating a claim. This
    is NOT a hallucination - it's the correct, safe behavior when retrieval
    didn't find enough - so it should be routed differently than a
    structural mismatch, not silently punished the same way.
    """
    return INSUFFICIENT_INFO_MARKER in response_text


def run_guardrail(
    query: str,
    context: List[Dict],
    generate_response_fn: Callable[[str, List[Dict]], str],
    config: Optional[GuardrailConfig] = None,
    refine_query_fn: Optional[Callable[[str, List[Dict]], str]] = None,
) -> GuardrailResult:
    """
    The main guardrail loop, implementing the algorithm's 5 steps with a
    bounded retry mechanism (steps 1-4 repeat on mismatch, up to
    max_refinement_attempts, exactly like D1's bounded retrieval loop).

    generate_response_fn(query, context) -> response_text : produces
        R_candidate (step 1) - typically Phase 2's CoT reasoning call.
    refine_query_fn(query, mismatch_details) -> refined_query : produces
        the next query when mismatch_score > 0 (step 5's refinement
        trigger) - can reuse Phase 2's C3 refine_query.
    """
    config = config or GuardrailConfig()
    attempts = []
    current_query = query
    current_context = context

    # Step 2: Extract Triples from Context (Graph_C) - built once, context
    # doesn't change across refinement attempts unless the query itself changes
    # what gets retrieved (handled by refine_query_fn producing a new query
    # for the CALLER to re-retrieve with, if they choose to).
    graph_c = build_context_graph(current_context)

    for attempt_num in range(1, config.max_refinement_attempts + 1):
        if config.verbose:
            print(f"\n[GUARDRAIL ATTEMPT {attempt_num}] Query: {current_query!r}")

        # Step 1: Generate Candidate Response
        r_candidate = generate_response_fn(current_query, current_context)

        # Honest "I don't have enough context" is not a hallucination -
        # route it as its own outcome rather than running triple alignment
        # on a response that isn't making groundable claims in the first place.
        if is_insufficient_info_response(r_candidate):
            if config.verbose:
                print(f"[GUARDRAIL] Model reported insufficient context - "
                      f"not a hallucination, but not answerable from this retrieval either.")
            attempts.append({
                "attempt": attempt_num,
                "query_used": current_query,
                "response": r_candidate,
                "mismatch_score": None,
                "aligned_triples": None,
                "total_triples": None,
            })
            if refine_query_fn and attempt_num < config.max_refinement_attempts:
                current_query = refine_query_fn(current_query, current_context)
                continue
            return GuardrailResult(
                query=query, verified=False, final_response=r_candidate,
                final_mismatch_score=None, attempts=attempts,
                stopped_reason="insufficient_info",
            )

        # Step 3: Extract Triples from Response (Graph_R)
        graph_r = build_response_graph(r_candidate)

        # Step 4: Perform Alignment Check
        alignment = compute_structural_alignment(
            graph_r, graph_c, similarity_threshold=config.similarity_threshold
        )

        if config.verbose:
            print_alignment_result(alignment)

        attempts.append({
            "attempt": attempt_num,
            "query_used": current_query,
            "response": r_candidate,
            "mismatch_score": alignment.mismatch_score,
            "aligned_triples": alignment.aligned_count,
            "total_triples": alignment.total_response_triples,
        })

        # Step 5: Decision - three tiers, not just binary:
        #   1. Fully aligned (mismatch == 0)              -> verified
        #   2. Within max_acceptable_mismatch but > 0      -> partially_verified
        #   3. Above max_acceptable_mismatch                -> refine/reject
        if alignment.is_fully_aligned:
            if config.verbose:
                print(f"[GUARDRAIL] Mismatch score = 0 -> response VERIFIED.")
            return GuardrailResult(
                query=query, verified=True, final_response=r_candidate,
                final_mismatch_score=alignment.mismatch_score,
                attempts=attempts, stopped_reason="verified",
                partially_verified=False, unverified_claims=[],
            )

        if config.max_acceptable_mismatch > 0 and alignment.mismatch_score <= config.max_acceptable_mismatch:
            if config.verbose:
                print(f"[GUARDRAIL] Mismatch score = {alignment.mismatch_score:.3f} is within the "
                      f"accepted tolerance ({config.max_acceptable_mismatch}) -> response PARTIALLY VERIFIED. "
                      f"{alignment.mismatched_count} unverified claim(s) flagged for transparency, "
                      f"not silently dropped.")
            return GuardrailResult(
                query=query, verified=True, final_response=r_candidate,
                final_mismatch_score=alignment.mismatch_score,
                attempts=attempts, stopped_reason="verified",
                partially_verified=True, unverified_claims=alignment.mismatch_details(),
            )

        # Mismatch detected beyond tolerance - trigger refinement (if a refine function is available)
        if config.verbose:
            print(f"[GUARDRAIL] Mismatch score = {alignment.mismatch_score:.3f} -> "
                  f"triggering refinement (attempt {attempt_num}/{config.max_refinement_attempts}).")

        if refine_query_fn and attempt_num < config.max_refinement_attempts:
            mismatch_details = alignment.mismatch_details()
            current_query = refine_query_fn(current_query, current_context)

    # Exhausted all attempts without reaching an acceptable mismatch level
    last_attempt = attempts[-1] if attempts else None
    return GuardrailResult(
        query=query, verified=False,
        final_response=last_attempt["response"] if last_attempt else None,
        final_mismatch_score=last_attempt["mismatch_score"] if last_attempt else None,
        attempts=attempts, stopped_reason="max_attempts_reached",
        partially_verified=False, unverified_claims=[],
    )


def print_guardrail_result(result: GuardrailResult):
    print(f"\n{'#'*60}")
    print(f"Query: {result.query}")
    print(f"Verified: {result.verified} (stopped: {result.stopped_reason})")
    print(f"Attempts used: {len(result.attempts)}")
    print(f"Final mismatch score: {result.final_mismatch_score}")
    if result.verified and not result.partially_verified:
        print(f"\nVERIFIED RESPONSE:\n{result.final_response}")
    elif result.verified and result.partially_verified:
        print(f"\n[PARTIALLY VERIFIED] Response accepted with "
              f"{len(result.unverified_claims)} unverified claim(s) - shown below for transparency, "
              f"NOT silently included as if fully grounded:")
        for claim in result.unverified_claims:
            print(f"  [UNVERIFIED] {claim.response_triple}")
        print(f"\nRESPONSE:\n{result.final_response}")
    elif result.stopped_reason == "insufficient_info":
        print(f"\n[INSUFFICIENT CONTEXT] The model honestly reported it could not answer "
              f"from the retrieved context after {len(result.attempts)} attempts - "
              f"NOT a detected hallucination, just a retrieval/coverage gap for this query.")
    else:
        print(f"\n[REJECTED - HALLUCINATION SUSPECTED] Response failed structural alignment "
              f"after {len(result.attempts)} attempts. Last candidate response withheld from the user.")
    print("#" * 60)


# --- Self-test: full algorithm, no external LLM/API needed ---
if __name__ == "__main__":
    fake_context = [
        {"verse_key": "21:83", "source_type": "tafsir", "text": "أيوب صبر على البلاء وشكر الله"},
    ]

    print("[TEST 1] A grounded response should verify on the first attempt...")
    def good_response_fn(query, context):
        return "أيوب صبر على البلاء"  # directly echoes the context - should align well

    result1 = run_guardrail(
        query="من صبر على البلاء",
        context=fake_context,
        generate_response_fn=good_response_fn,
        config=GuardrailConfig(verbose=True),
    )
    print_guardrail_result(result1)
    assert result1.verified, "Expected a grounded response to pass verification"
    assert result1.stopped_reason == "verified"
    print("\n[PASS] Test 1 passed.\n")

    print("[TEST 2] A hallucinating LLM should get rejected after exhausting refinement attempts...")
    def bad_response_fn(query, context):
        return "موسى بنى الكعبة في مكة"  # unrelated fabrication, every attempt

    def dummy_refine_fn(query, context):
        return query + " (refined)"

    result2 = run_guardrail(
        query="من صبر على البلاء",
        context=fake_context,
        generate_response_fn=bad_response_fn,
        refine_query_fn=dummy_refine_fn,
        config=GuardrailConfig(max_refinement_attempts=2, verbose=True),
    )
    print_guardrail_result(result2)
    assert not result2.verified, "Expected a consistently hallucinating response to be rejected"
    assert result2.stopped_reason == "max_attempts_reached"
    assert len(result2.attempts) == 2, "Expected exactly max_refinement_attempts attempts"
    print("\n[PASS] Test 2 passed.\n")

    print("[TEST 3] A response that improves after refinement should verify on attempt 2...")
    call_count = {"n": 0}
    def improving_response_fn(query, context):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "فرعون طغى في الارض"  # wrong on first try
        return "أيوب صبر على البلاء"      # correct on retry

    result3 = run_guardrail(
        query="من صبر على البلاء",
        context=fake_context,
        generate_response_fn=improving_response_fn,
        refine_query_fn=dummy_refine_fn,
        config=GuardrailConfig(max_refinement_attempts=3, verbose=True),
    )
    print_guardrail_result(result3)
    assert result3.verified, "Expected the improved second attempt to pass"
    assert len(result3.attempts) == 2, "Expected verification to happen on attempt 2, not fewer/more"
    print("\n[PASS] Test 3 passed.\n")

    print("[TEST 4] Model honestly reports insufficient context - should NOT be treated as hallucination...")
    def insufficient_response_fn(query, context):
        return "Reasoning:\nلا يوجد ذكر لهذا في السياق المقدم.\n\nAnswer:\nغير كاف"

    result4 = run_guardrail(
        query="سؤال غامض جدا",
        context=fake_context,
        generate_response_fn=insufficient_response_fn,
        refine_query_fn=dummy_refine_fn,
        config=GuardrailConfig(max_refinement_attempts=2, verbose=True),
    )
    print_guardrail_result(result4)
    assert not result4.verified
    assert result4.stopped_reason == "insufficient_info", \
        f"Expected 'insufficient_info', got '{result4.stopped_reason}'"
    assert result4.final_mismatch_score is None, \
        "Insufficient-info responses should not get a mismatch score at all"
    print("\n[PASS] Test 4 passed.\n")

    print("[TEST 5] Partial verification: mostly-grounded response accepted with tolerance...")
    def mostly_grounded_response_fn(query, context):
        # Real claim + one fabricated add-on - should be ~50% mismatched
        return "أيوب صبر على البلاء. وأيضا فرعون بنى الاهرامات."

    result5 = run_guardrail(
        query="من صبر على البلاء",
        context=fake_context,
        generate_response_fn=mostly_grounded_response_fn,
        config=GuardrailConfig(max_refinement_attempts=1, max_acceptable_mismatch=0.6, verbose=True),
    )
    print_guardrail_result(result5)
    assert result5.verified and result5.partially_verified, \
        "Expected a mostly-grounded response to be accepted as partially verified under tolerance"
    assert len(result5.unverified_claims) > 0, "Expected the unverified claim to be reported, not hidden"
    print("\n[PASS] Test 5 passed.\n")

    print("[RESULT] All E4 self-tests passed - full guardrail algorithm verified end-to-end.")
