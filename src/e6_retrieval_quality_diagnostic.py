"""
E6 - Retrieval Quality Diagnostic.

Phase 3's guardrail can only verify a response against WHATEVER context got
retrieved - it has no way to know if that context was actually relevant to
the query in the first place. This script fills that gap: it runs a batch
of queries through Phase 1's real retrieval_api and flags any query whose
top similarity scores are suspiciously low, so weak retrieval gets caught
and reported to whoever owns Phase 1/embedding quality, rather than being
silently blamed on the guardrail or the LLM.

This is diagnostic tooling, not a guardrail component - it doesn't change
any Phase 3 decision, it just tells you WHERE to look when results seem off.

Requirements:
    (uses whatever retrieval_api Phase 1 already provides - no new deps)
"""

import sys
from dataclasses import dataclass, field
from typing import List, Dict, Callable


@dataclass
class RetrievalQualityReport:
    query: str
    top_similarity: float
    avg_similarity: float
    is_low_quality: bool
    retrieved_verse_keys: List[str] = field(default_factory=list)


def diagnose_retrieval_quality(
    retrieval_api,
    queries: List[str],
    top_k: int = 5,
    low_quality_threshold: float = 0.5,
) -> List[RetrievalQualityReport]:
    """
    Run each query through retrieval_api.retrieve() and flag ones where even
    the TOP result's similarity is below low_quality_threshold - a strong
    signal that the embedding model/index didn't find anything genuinely
    relevant, independent of whatever the LLM or guardrail does afterward.
    """
    reports = []
    for query in queries:
        results = retrieval_api.retrieve(query, top_k=top_k)
        similarities = [r.get("similarity", 0.0) for r in results]
        top_sim = similarities[0] if similarities else 0.0
        avg_sim = sum(similarities) / len(similarities) if similarities else 0.0

        reports.append(RetrievalQualityReport(
            query=query,
            top_similarity=top_sim,
            avg_similarity=avg_sim,
            is_low_quality=top_sim < low_quality_threshold,
            retrieved_verse_keys=[r.get("verse_key", "?") for r in results],
        ))
    return reports


def print_retrieval_quality_report(reports: List[RetrievalQualityReport]):
    print(f"\n{'='*70}\nRETRIEVAL QUALITY DIAGNOSTIC\n{'='*70}")
    low_quality = [r for r in reports if r.is_low_quality]

    for r in reports:
        flag = "LOW QUALITY - flag for Phase 1 review" if r.is_low_quality else "OK"
        print(f"\n[{flag}] {r.query}")
        print(f"  top_similarity={r.top_similarity:.3f}, avg_similarity={r.avg_similarity:.3f}")
        print(f"  retrieved: {r.retrieved_verse_keys}")

    print(f"\n{'-'*70}")
    print(f"Summary: {len(low_quality)}/{len(reports)} queries flagged as low-quality retrieval.")
    if low_quality:
        print("These queries are worth reviewing with whoever owns Phase 1's embedding "
              "model/index - Phase 3's guardrail correctly rejects/flags answers built on "
              "weak context, but the ROOT problem for these specific queries is retrieval, "
              "not the guardrail or the LLM.")
        print("\nFlagged queries:")
        for r in low_quality:
            print(f"  - {r.query} (top_similarity={r.top_similarity:.3f})")
    print("=" * 70)


# --- Self-test with a fake retrieval API ---
if __name__ == "__main__":
    class FakeRetrievalAPI:
        """Simulates mixed retrieval quality: some queries get strong matches,
        some get weak/irrelevant ones - exactly the real pattern observed."""
        def retrieve(self, query, top_k=5):
            if "صبر" in query:  # patience - simulate GOOD retrieval
                return [
                    {"verse_key": "21:83", "similarity": 0.85, "text": "أيوب صبر"},
                    {"verse_key": "2:153", "similarity": 0.78, "text": "استعينوا بالصبر"},
                ]
            else:  # simulate WEAK/irrelevant retrieval, like "العدل" in the real run
                return [
                    {"verse_key": "36:2", "similarity": 0.35, "text": "والقرءان الحكيم"},
                    {"verse_key": "80:13", "similarity": 0.28, "text": "في صحف مكرمة"},
                ]

    api = FakeRetrievalAPI()
    test_queries = ["ما فوائد الصبر", "العدل في الإسلام"]

    print("[TEST] Running retrieval quality diagnostic on mixed-quality queries...")
    reports = diagnose_retrieval_quality(api, test_queries, low_quality_threshold=0.5)
    print_retrieval_quality_report(reports)

    assert not reports[0].is_low_quality, "Expected the patience query to score as good quality"
    assert reports[1].is_low_quality, "Expected the justice query to be flagged as low quality"
    print("\n[PASS] E6 self-test passed: correctly distinguished good vs. weak retrieval.")
