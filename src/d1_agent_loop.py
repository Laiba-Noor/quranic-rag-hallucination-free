"""
D1 - Autonomous Agent Loop Controller.

Wraps Phase 1's B6 RetrievalAPI in a bounded, iterative loop: retrieve,
evaluate sufficiency, and if insufficient, refine the query and retrieve
again - up to a hard maximum number of rounds so an unanswerable query can
never loop forever.

Sufficiency checking and query refinement are BOTH pluggable (dependency
injection), not hardcoded:
    - sufficiency_fn(query, results) -> (is_sufficient: bool, score: float)
      Defaults to a simple similarity-threshold heuristic. D3 replaces this
      with a threshold calibrated on Roma's C2 sufficiency-labeled dataset.
    - refine_fn(query, results) -> refined_query: str
      Defaults to a no-op placeholder that logs clearly it isn't real
      refinement yet. Roma's C3 query-refinement logic plugs in here.

Requirements:
    pip install sentence-transformers hnswlib pandas   (via Phase 1's B6)
"""

import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional, Tuple


@dataclass
class AgentConfig:
    max_iterations: int = 3
    top_k: int = 5
    default_sufficiency_threshold: float = 0.55  # placeholder until D3 calibrates a real one
    verbose: bool = True


@dataclass
class IterationLog:
    round_number: int
    query_used: str
    retrieved: List[Dict]
    is_sufficient: bool
    sufficiency_score: float


@dataclass
class AgentResult:
    original_query: str
    final_query: str
    final_context: List[Dict]
    iterations: List[IterationLog]
    stopped_reason: str  # "sufficient" | "max_iterations_reached"


def default_sufficiency_fn(query: str, results: List[Dict],
                            threshold: float = 0.55) -> Tuple[bool, float]:
    """
    Simple heuristic: sufficient if the top retrieved result's similarity
    score clears a fixed threshold. This is a placeholder - D3 replaces it
    with a threshold calibrated against Roma's C2 labeled dataset, which is
    a real signal instead of a guessed constant.
    """
    if not results:
        return False, 0.0
    top_score = results[0].get("similarity", 0.0)
    return top_score >= threshold, top_score


def default_refine_fn(query: str, results: List[Dict]) -> str:
    """
    Placeholder query refinement: logs clearly that no real refinement logic
    is wired in yet, and returns the query unchanged. Roma's C3 replaces
    this so the loop actually improves its second attempt instead of
    repeating the same failed query.
    """
    print("[WARN] Using placeholder refine_fn - query left unchanged. "
          "Wire in Roma's C3 query-refinement logic (see D3) for real retries.")
    return query


class AgenticRetriever:
    """
    The bounded retrieval loop. Call .run(query) to get an AgentResult with
    the final context and a full per-round log.
    """

    def __init__(
        self,
        retrieval_api,  # an instance of Phase 1's B6 RetrievalAPI
        config: Optional[AgentConfig] = None,
        sufficiency_fn: Optional[Callable[[str, List[Dict]], Tuple[bool, float]]] = None,
        refine_fn: Optional[Callable[[str, List[Dict]], str]] = None,
    ):
        self.retrieval_api = retrieval_api
        self.config = config or AgentConfig()
        self.sufficiency_fn = sufficiency_fn or (
            lambda q, r: default_sufficiency_fn(q, r, self.config.default_sufficiency_threshold)
        )
        self.refine_fn = refine_fn or default_refine_fn

    def run(self, query: str) -> AgentResult:
        current_query = query
        iterations = []
        final_results = []
        stopped_reason = "max_iterations_reached"

        for round_number in range(1, self.config.max_iterations + 1):
            if self.config.verbose:
                print(f"\n[ROUND {round_number}] Query: {current_query!r}")

            results = self.retrieval_api.retrieve(current_query, top_k=self.config.top_k)
            is_sufficient, score = self.sufficiency_fn(current_query, results)

            iterations.append(IterationLog(
                round_number=round_number,
                query_used=current_query,
                retrieved=results,
                is_sufficient=is_sufficient,
                sufficiency_score=score,
            ))

            if self.config.verbose:
                print(f"  Retrieved {len(results)} results, "
                      f"sufficiency_score={score:.3f}, sufficient={is_sufficient}")

            final_results = results

            if is_sufficient:
                stopped_reason = "sufficient"
                break

            if round_number < self.config.max_iterations:
                current_query = self.refine_fn(current_query, results)

        return AgentResult(
            original_query=query,
            final_query=current_query,
            final_context=final_results,
            iterations=iterations,
            stopped_reason=stopped_reason,
        )


def print_agent_result(result: AgentResult):
    print(f"\n{'='*60}")
    print(f"Original query : {result.original_query}")
    print(f"Final query    : {result.final_query}")
    print(f"Rounds used    : {len(result.iterations)}")
    print(f"Stopped because: {result.stopped_reason}")
    print(f"Final context  : {len(result.final_context)} items")
    for item in result.final_context:
        print(f"  [{item.get('source_type')}] {item.get('verse_key')} "
              f"(sim={item.get('similarity', 0):.3f})")
    print("=" * 60)


# --- Self-test with a fake retrieval API, no Phase 1 dependency needed ---
if __name__ == "__main__":
    class FakeRetrievalAPI:
        """Stands in for Phase 1's B6 RetrievalAPI so D1 can be tested standalone."""
        def __init__(self):
            self.call_count = 0

        def retrieve(self, query, top_k=5):
            self.call_count += 1
            if self.call_count == 1:
                return [{"verse_key": "2:100", "source_type": "verse", "similarity": 0.3,
                          "text": "weak match"}]
            return [{"verse_key": "1:1", "source_type": "verse", "similarity": 0.9,
                      "text": "strong match"}]

    def test_refine_fn(query, results):
        return query + " (refined)"

    print("[TEST] Running AgenticRetriever with a fake retrieval API...")
    agent = AgenticRetriever(
        retrieval_api=FakeRetrievalAPI(),
        config=AgentConfig(max_iterations=3, verbose=True),
        refine_fn=test_refine_fn,
    )
    result = agent.run("test query about mercy")
    print_agent_result(result)

    assert result.stopped_reason == "sufficient", "Expected loop to succeed on round 2"
    assert len(result.iterations) == 2, "Expected exactly 2 rounds (weak then strong)"
    print("\n[PASS] D1 self-test passed: loop retried once and stopped on sufficient context.")
