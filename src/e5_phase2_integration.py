"""
E5 - Phase 2 -> Phase 3 Integration.

Wraps Phase 2's TheologicalAgent so its CoT-generated response is passed
through Phase 3's zero-hallucination guardrail before being returned to the
user. This is the "verified answer" endpoint the whole pipeline has been
building toward: retrieval (Phase 1) -> agentic reasoning (Phase 2) ->
structural verification (Phase 3).

Design: Phase 2's TheologicalAgent already does retrieval + CoT reasoning
+ cross-lingual fallback. This wrapper takes its output and:
    1. Uses its retrieved context as Graph_C's source (already real,
       already retrieved - no extra retrieval work).
    2. Uses its CoT answer as R_candidate for Graph_R.
    3. Runs the guardrail's alignment check.
    4. On mismatch, reuses Phase 2's own refine_query (C3) to get a new
       query and re-runs the whole TheologicalAgent.answer() call - so
       refinement genuinely re-retrieves, not just re-prompts the same
       stale context.
"""

import sys
from dataclasses import dataclass
from typing import Optional

from e4_zero_hallucination_guardrail import run_guardrail, GuardrailConfig, GuardrailResult, print_guardrail_result


def verify_theological_agent_answer(
    theological_agent,   # Phase 2's TheologicalAgent instance
    query: str,
    refine_query_fn=None,  # Phase 2's C3 refine_query, or None to disable refinement
    guardrail_config: Optional[GuardrailConfig] = None,
) -> GuardrailResult:
    """
    Run Phase 2's agent, then verify its answer through Phase 3's guardrail.
    If the guardrail rejects it, re-runs the FULL Phase 2 agent (fresh
    retrieval) on the refined query, up to max_refinement_attempts.
    """
    config = guardrail_config or GuardrailConfig()

    def generate_response_fn(q, context):
        # context is ignored here in favor of re-running the real agent,
        # since Phase 2's agent does its own retrieval - this closure just
        # needs to match E4's expected signature.
        phase2_result = theological_agent.answer(q)
        generate_response_fn.last_phase2_result = phase2_result  # stash for context access
        if phase2_result.cot_result:
            return phase2_result.cot_result.answer
        return ""

    def get_current_context():
        result = getattr(generate_response_fn, "last_phase2_result", None)
        return result.agent_result.final_context if result else []

    # Run once to get real, fresh context for Graph_C
    initial_response = generate_response_fn(query, [])
    real_context = get_current_context()

    return run_guardrail(
        query=query,
        context=real_context,
        generate_response_fn=lambda q, c: (
            generate_response_fn(q, c) if q != query else initial_response
        ),
        refine_query_fn=refine_query_fn,
        config=config,
    )


# --- Self-test using a fake TheologicalAgent, no Phase 1/2 dependencies needed ---
if __name__ == "__main__":
    class FakeCoTResult:
        def __init__(self, answer):
            self.answer = answer

    class FakeAgentResult:
        def __init__(self, context):
            self.final_context = context

    class FakePhase2Result:
        def __init__(self, answer, context):
            self.cot_result = FakeCoTResult(answer)
            self.agent_result = FakeAgentResult(context)

    class FakeTheologicalAgent:
        """Simulates Phase 2's agent: always retrieves the same context, but
        the answer quality can vary to test the guardrail's reaction."""
        def __init__(self, answers):
            self.answers = answers
            self.call_count = 0

        def answer(self, query):
            context = [{"verse_key": "21:83", "source_type": "tafsir",
                        "text": "أيوب صبر على البلاء وشكر الله"}]
            answer_text = self.answers[min(self.call_count, len(self.answers) - 1)]
            self.call_count += 1
            return FakePhase2Result(answer_text, context)

    def dummy_refine(query, context):
        return query + " (refined)"

    print("[TEST 1] Phase 2 gives a grounded answer immediately...")
    agent1 = FakeTheologicalAgent(answers=["أيوب صبر على البلاء"])
    result1 = verify_theological_agent_answer(
        agent1, "من صبر على البلاء", refine_query_fn=dummy_refine,
        guardrail_config=GuardrailConfig(verbose=True),
    )
    print_guardrail_result(result1)
    assert result1.verified
    print("\n[PASS] Test 1 passed.\n")

    print("[TEST 2] Phase 2 hallucinates first, then gives a grounded answer on refinement...")
    agent2 = FakeTheologicalAgent(answers=["فرعون طغى في الارض", "أيوب صبر على البلاء"])
    result2 = verify_theological_agent_answer(
        agent2, "من صبر على البلاء", refine_query_fn=dummy_refine,
        guardrail_config=GuardrailConfig(max_refinement_attempts=3, verbose=True),
    )
    print_guardrail_result(result2)
    assert result2.verified
    assert len(result2.attempts) == 2
    print("\n[PASS] Test 2 passed.\n")

    print("[RESULT] All E5 integration self-tests passed - Phase 2 -> Phase 3 wiring confirmed.")
