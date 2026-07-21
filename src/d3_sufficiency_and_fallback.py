"""
D3 - Sufficiency Calibration + Cross-Lingual Fallback Wiring.

Two real, usable pieces (not stubs) plus the final assembly:

1. SufficiencyScorer - starts with D1's fixed-threshold heuristic, but adds
   a genuine calibrate() method: given Roma's C2 labeled dataset
   [{"top_similarity": float, "is_sufficient": bool}, ...], it searches for
   the threshold that maximizes accuracy. This replaces the guessed
   constant in D1 with an evidence-based one the moment C2 is delivered.

2. CrossLingualFallback - a clear, honest interface for Roma's C1 resources.
   Until C1 lands, it runs in "no-op" mode and says so explicitly in its
   output rather than pretending to verify something it can't.

3. TheologicalAgent - the full pipeline: D1's loop + D2's CoT reasoning +
   this file's calibrated sufficiency and fallback, wired together as the
   single object Phase 2's testing (and eventually Phase 3) calls.
"""

import sys
from dataclasses import dataclass
from typing import Callable, List, Dict, Optional, Tuple

from d1_agent_loop import AgenticRetriever, AgentConfig, AgentResult, print_agent_result
from d2_cot_prompting import run_cot_reasoning, CoTResult, print_cot_result


class SufficiencyScorer:
    """
    Wraps a threshold-based sufficiency check. Starts with a reasonable
    default; calibrate() replaces it with a real, evidence-based threshold
    once Roma's C2 labeled dataset is available.
    """

    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold
        self.calibrated = False

    def __call__(self, query: str, results: List[Dict]) -> Tuple[bool, float]:
        if not results:
            return False, 0.0
        top_score = results[0].get("similarity", 0.0)
        return top_score >= self.threshold, top_score

    def calibrate(self, labeled_examples: List[Dict]) -> float:
        """
        labeled_examples: [{"top_similarity": float, "is_sufficient": bool}, ...]
        from Roma's C2 dataset. Searches candidate thresholds and picks the
        one maximizing accuracy against the labeled examples.
        """
        if not labeled_examples:
            print("[WARN] No labeled examples provided - keeping default threshold "
                  f"({self.threshold}).")
            return self.threshold

        candidate_thresholds = sorted({round(e["top_similarity"], 3) for e in labeled_examples})
        best_threshold = self.threshold
        best_accuracy = -1.0

        for t in candidate_thresholds:
            correct = sum(
                1 for e in labeled_examples
                if (e["top_similarity"] >= t) == e["is_sufficient"]
            )
            accuracy = correct / len(labeled_examples)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = t

        self.threshold = best_threshold
        self.calibrated = True
        print(f"[OK] Calibrated sufficiency threshold: {best_threshold:.3f} "
              f"(accuracy={best_accuracy:.1%} on {len(labeled_examples)} labeled examples)")
        return best_threshold


class CrossLingualFallback:
    """
    Verification-only fallback: routes an ambiguous query through Roma's C1
    resources (AyaTEC + SQuAD v2) to cross-check reasoning before answering
    back in Arabic. NEVER the primary answer source - only invoked when the
    agent's confidence is low even after retries.
    """

    def __init__(self, lookup_fn: Optional[Callable[[str], Optional[Dict]]] = None):
        self.lookup_fn = lookup_fn
        self.enabled = lookup_fn is not None

    def verify(self, query: str) -> Dict:
        if not self.enabled:
            return {
                "used": False,
                "reason": "no_op - Roma's C1 fallback resources not wired in yet",
                "verification": None,
            }

        result = self.lookup_fn(query)
        return {
            "used": True,
            "reason": "cross-lingual verification attempted",
            "verification": result,
        }


@dataclass
class TheologicalAgentResult:
    agent_result: AgentResult
    cot_result: Optional[CoTResult]
    fallback_result: Optional[Dict]
    used_fallback: bool


class TheologicalAgent:
    """
    The complete Phase 2 pipeline: D1's bounded retrieval loop, D2's
    Chain-of-Thought reasoning, and this file's calibrated sufficiency +
    cross-lingual fallback - all wired together as one callable object.
    """

    def __init__(
        self,
        retrieval_api,
        call_llm_fn: Callable[[str], str],
        config: Optional[AgentConfig] = None,
        sufficiency_scorer: Optional[SufficiencyScorer] = None,
        fallback: Optional[CrossLingualFallback] = None,
        refine_fn: Optional[Callable[[str, List[Dict]], str]] = None,
        low_confidence_fallback_threshold: float = 0.35,
        max_chars_per_context_item: int = 400,
    ):
        self.sufficiency_scorer = sufficiency_scorer or SufficiencyScorer()
        self.fallback = fallback or CrossLingualFallback()
        self.call_llm_fn = call_llm_fn
        self.low_confidence_fallback_threshold = low_confidence_fallback_threshold
        self.max_chars_per_context_item = max_chars_per_context_item

        self.retriever = AgenticRetriever(
            retrieval_api=retrieval_api,
            config=config or AgentConfig(),
            sufficiency_fn=self.sufficiency_scorer,
            refine_fn=refine_fn,
        )

    def answer(self, query: str) -> TheologicalAgentResult:
        agent_result = self.retriever.run(query)

        final_score = agent_result.iterations[-1].sufficiency_score if agent_result.iterations else 0.0
        used_fallback = final_score < self.low_confidence_fallback_threshold
        fallback_result = None

        if used_fallback:
            print(f"[INFO] Final confidence ({final_score:.3f}) is below the fallback "
                  f"threshold ({self.low_confidence_fallback_threshold}) - "
                  f"attempting cross-lingual verification.")
            fallback_result = self.fallback.verify(query)

        cot_result = None
        if agent_result.final_context:
            cot_result = run_cot_reasoning(
                query=agent_result.final_query,
                context=agent_result.final_context,
                call_llm_fn=self.call_llm_fn,
                max_chars_per_item=self.max_chars_per_context_item,
            )

        return TheologicalAgentResult(
            agent_result=agent_result,
            cot_result=cot_result,
            fallback_result=fallback_result,
            used_fallback=used_fallback,
        )


def print_theological_agent_result(result: TheologicalAgentResult):
    print_agent_result(result.agent_result)
    if result.used_fallback:
        print(f"\n[FALLBACK] {result.fallback_result}")
    if result.cot_result:
        print_cot_result(result.cot_result)


# --- Self-test: full pipeline with fake retrieval + fake LLM, no real deps ---
if __name__ == "__main__":
    class FakeRetrievalAPI:
        def retrieve(self, query, top_k=5):
            return [
                {"verse_key": "1:1", "source_type": "verse", "similarity": 0.9,
                 "text": "In the name of Allah, the Most Merciful"},
                {"verse_key": "1:2", "source_type": "tafsir", "similarity": 0.8,
                 "text": "This establishes Allah's mercy as foundational"},
            ]

    def fake_llm(prompt: str) -> str:
        return (
            "Reasoning:\nThe text opens by invoking mercy directly [1:1], "
            "and commentary confirms its centrality [1:2].\n\n"
            "Answer:\nMercy is a foundational theme, established immediately [1:1][1:2]."
        )

    print("[TEST 1] Calibrating SufficiencyScorer on fake labeled data...")
    scorer = SufficiencyScorer(threshold=0.5)
    labeled = [
        {"top_similarity": 0.9, "is_sufficient": True},
        {"top_similarity": 0.8, "is_sufficient": True},
        {"top_similarity": 0.3, "is_sufficient": False},
        {"top_similarity": 0.2, "is_sufficient": False},
        {"top_similarity": 0.6, "is_sufficient": True},
    ]
    scorer.calibrate(labeled)
    assert scorer.calibrated
    assert 0.3 < scorer.threshold <= 0.6, f"Expected a sensible threshold, got {scorer.threshold}"
    print("[PASS] Calibration found a threshold separating the labeled classes.\n")

    print("[TEST 2] Running the full TheologicalAgent pipeline (no fallback needed)...")
    agent = TheologicalAgent(
        retrieval_api=FakeRetrievalAPI(),
        call_llm_fn=fake_llm,
        sufficiency_scorer=scorer,
        config=AgentConfig(max_iterations=2, verbose=True),
    )
    result = agent.answer("What does the Quran say about mercy?")
    print_theological_agent_result(result)

    assert result.agent_result.stopped_reason == "sufficient"
    assert not result.used_fallback, "High-confidence result should not trigger fallback"
    assert result.cot_result.grounding_ok
    print("\n[PASS] D3 self-test 1 passed: full pipeline ran without needing fallback.")

    print("\n[TEST 3] Testing fallback trigger on a low-confidence case...")
    class WeakRetrievalAPI:
        def retrieve(self, query, top_k=5):
            return [{"verse_key": "5:5", "source_type": "verse", "similarity": 0.1,
                      "text": "unrelated weak match"}]

    weak_agent = TheologicalAgent(
        retrieval_api=WeakRetrievalAPI(),
        call_llm_fn=fake_llm,
        config=AgentConfig(max_iterations=1, verbose=False),
    )
    weak_result = weak_agent.answer("an obscure ambiguous query")
    assert weak_result.used_fallback, "Low-confidence result should trigger fallback"
    assert weak_result.fallback_result["used"] is False, "Fallback should run in no-op mode (no C1 wired in yet)"
    print(f"[OK] Fallback triggered correctly: {weak_result.fallback_result}")
    print("\n[PASS] D3 self-test 3 passed: fallback correctly triggers and runs safely in no-op mode.")
