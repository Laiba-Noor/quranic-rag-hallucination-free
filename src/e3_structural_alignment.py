"""
E3 - Structural Alignment Filtering.

Compares Graph_R (from the candidate response) against Graph_C (from the
retrieved context) and computes a Mismatch_Score: for every triple in the
response, is there a structurally similar triple somewhere in the context?
Triples with no good match are exactly the claims a hallucination guardrail
should flag - content in the answer that isn't actually traceable back to
what was retrieved.

Matching uses fuzzy string similarity (rapidfuzz) across subject, relation,
and object jointly, rather than requiring exact string equality - real
Arabic text varies in phrasing even when saying the same thing, so exact
match would produce false positives (flagging correct claims as
hallucinated) far too often to be usable.

Requirements:
    pip install rapidfuzz networkx
"""

import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import networkx as nx
from rapidfuzz import fuzz

from e1_triple_extraction import Triple
from e2_knowledge_graph import build_graph, build_context_graph, build_response_graph


@dataclass
class TripleMatch:
    response_triple: Tuple[str, str, str]
    best_context_match: Optional[Tuple[str, str, str]]
    similarity_score: float  # 0-100
    aligned: bool


@dataclass
class AlignmentResult:
    mismatch_score: float  # 0.0 = perfectly aligned, 1.0 = nothing aligned
    total_response_triples: int
    aligned_count: int
    mismatched_count: int
    matches: List[TripleMatch] = field(default_factory=list)

    @property
    def is_fully_aligned(self) -> bool:
        return self.mismatch_score == 0.0

    def mismatch_details(self) -> List[TripleMatch]:
        """The specific unaligned triples - exactly what a refinement loop needs as feedback."""
        return [m for m in self.matches if not m.aligned]


def graph_to_triple_list(graph: nx.MultiDiGraph) -> List[Tuple[str, str, str]]:
    return [(u, data.get("relation", ""), v) for u, v, data in graph.edges(data=True)]


def triple_similarity(t1: Tuple[str, str, str], t2: Tuple[str, str, str]) -> float:
    """
    Combined fuzzy similarity across subject, relation, object (0-100).
    Relation gets slightly more weight - two claims about the same
    entities but with a contradicted relation (e.g. "forbade" vs
    "permitted") should NOT be treated as aligned just because the
    subject/object text matches.
    """
    subj_sim = fuzz.token_set_ratio(t1[0], t2[0])
    rel_sim = fuzz.ratio(t1[1], t2[1])
    obj_sim = fuzz.token_set_ratio(t1[2], t2[2])
    return 0.3 * subj_sim + 0.4 * rel_sim + 0.3 * obj_sim


def compute_structural_alignment(
    graph_r: nx.MultiDiGraph,
    graph_c: nx.MultiDiGraph,
    similarity_threshold: float = 65.0,
) -> AlignmentResult:
    """
    Implements the methodology's alignment check: for every triple in the
    response graph, find its best match in the context graph and decide if
    it clears the similarity threshold. Mismatch_Score is the fraction of
    response triples with NO adequate match - directly implementing
    "Compute_Structural_Distance(Graph_R, Graph_C)" from the algorithm.
    """
    response_triples = graph_to_triple_list(graph_r)
    context_triples = graph_to_triple_list(graph_c)

    matches = []
    for r_triple in response_triples:
        if not context_triples:
            matches.append(TripleMatch(r_triple, None, 0.0, aligned=False))
            continue

        best_score = -1.0
        best_match = None
        for c_triple in context_triples:
            score = triple_similarity(r_triple, c_triple)
            if score > best_score:
                best_score = score
                best_match = c_triple

        aligned = best_score >= similarity_threshold
        matches.append(TripleMatch(r_triple, best_match, best_score, aligned))

    total = len(matches)
    aligned_count = sum(1 for m in matches if m.aligned)
    mismatched_count = total - aligned_count
    mismatch_score = (mismatched_count / total) if total > 0 else 1.0  # no claims at all = can't verify = worst case

    return AlignmentResult(
        mismatch_score=mismatch_score,
        total_response_triples=total,
        aligned_count=aligned_count,
        mismatched_count=mismatched_count,
        matches=matches,
    )


def print_alignment_result(result: AlignmentResult):
    print(f"\n{'='*60}")
    print(f"Mismatch score: {result.mismatch_score:.3f} "
          f"({'FULLY ALIGNED' if result.is_fully_aligned else 'MISMATCH DETECTED'})")
    print(f"Aligned: {result.aligned_count}/{result.total_response_triples}")
    for m in result.matches:
        status = "OK" if m.aligned else "MISMATCH"
        print(f"  [{status}] (sim={m.similarity_score:.1f}) {m.response_triple}")
        if not m.aligned:
            print(f"      closest context match: {m.best_context_match}")
    print("=" * 60)


def calibrate_similarity_threshold(
    known_grounded_pairs: List[Tuple[nx.MultiDiGraph, nx.MultiDiGraph]],
    known_hallucinated_pairs: List[Tuple[nx.MultiDiGraph, nx.MultiDiGraph]],
    candidate_thresholds: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """
    Replace the guessed 65.0 similarity_threshold with an evidence-based one,
    the same way Roma's C2 SufficiencyScorer.calibrate() replaced Phase 2's
    guessed sufficiency threshold.

    known_grounded_pairs: [(graph_r, graph_c), ...] where graph_r SHOULD align
        with graph_c (real, correct responses paired with their real context).
    known_hallucinated_pairs: [(graph_r, graph_c), ...] where graph_r should
        NOT align (fabricated/wrong responses paired with unrelated context).

    Searches thresholds and picks the one that best separates the two sets
    by whether is_fully_aligned comes out correct - directly measurable
    ground truth, not a guess.

    Returns (best_threshold, accuracy_at_best_threshold).
    """
    if candidate_thresholds is None:
        candidate_thresholds = [float(t) for t in range(30, 96, 5)]  # 30, 35, ..., 95

    if not known_grounded_pairs and not known_hallucinated_pairs:
        print("[WARN] No labeled examples provided - cannot calibrate. Keeping default threshold.")
        return 65.0, 0.0

    best_threshold = 65.0
    best_accuracy = -1.0

    for threshold in candidate_thresholds:
        correct = 0
        total = 0

        for graph_r, graph_c in known_grounded_pairs:
            result = compute_structural_alignment(graph_r, graph_c, similarity_threshold=threshold)
            correct += int(result.is_fully_aligned)  # SHOULD be aligned
            total += 1

        for graph_r, graph_c in known_hallucinated_pairs:
            result = compute_structural_alignment(graph_r, graph_c, similarity_threshold=threshold)
            correct += int(not result.is_fully_aligned)  # SHOULD NOT be aligned
            total += 1

        accuracy = correct / total if total else 0.0
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    print(f"[OK] Calibrated similarity_threshold: {best_threshold} "
          f"(accuracy={best_accuracy:.1%} on {len(known_grounded_pairs)} grounded + "
          f"{len(known_hallucinated_pairs)} hallucinated labeled examples)")
    return best_threshold, best_accuracy


# --- Self-test ---
if __name__ == "__main__":
    print("[TEST 1] Fully aligned response (claims match context)...")
    context = [
        {"verse_key": "21:83", "source_type": "tafsir", "text": "أيوب صبر على البلاء"},
    ]
    graph_c = build_context_graph(context)
    graph_r = build_response_graph("أيوب صبر على البلاء")  # same claim, should align well
    result = compute_structural_alignment(graph_r, graph_c)
    print_alignment_result(result)
    assert result.mismatch_score < 0.5, "Expected a near-identical claim to align well"
    print("[PASS]\n")

    print("[TEST 2] Hallucinated response (claims NOT in context)...")
    graph_r2 = build_response_graph("موسى بنى الكعبة في مكة")  # fabricated, unrelated claim
    result2 = compute_structural_alignment(graph_r2, graph_c)
    print_alignment_result(result2)
    assert result2.mismatch_score > 0.5, "Expected an unrelated claim to be flagged as mismatched"
    print("[PASS]\n")

    print("[TEST 3] Calibrating the similarity threshold on labeled examples...")
    grounded_pairs = [
        (build_response_graph("أيوب صبر على البلاء"), graph_c),
        (build_response_graph("صبر أيوب على البلاء وشكر"), graph_c),
    ]
    hallucinated_pairs = [
        (build_response_graph("موسى بنى الكعبة في مكة"), graph_c),
        (build_response_graph("فرعون طغى في الارض وادعى الالوهية"), graph_c),
    ]
    best_t, acc = calibrate_similarity_threshold(grounded_pairs, hallucinated_pairs)
    print(f"  Best threshold: {best_t}, accuracy: {acc:.1%}")
    assert acc >= 0.75, f"Expected calibration to find a reasonably separating threshold, got {acc:.1%}"
    print("[PASS]\n")

    print("[RESULT] All E3 self-tests passed.")
