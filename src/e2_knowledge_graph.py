"""
E2 - Real-Time Knowledge Graph Construction.

Builds a directed graph from a list of (subject, relation, object) triples
using networkx: nodes are subjects/objects, edges are relations. Two graphs
get built per guardrail check - Graph_C (from retrieved context) and
Graph_R (from the LLM's candidate response) - which E3 then compares.

Requirements:
    pip install networkx
"""

import sys
from typing import List, Optional
import networkx as nx

from e1_triple_extraction import Triple, extract_triples, extract_triples_from_context


def build_graph(triples: List[Triple]) -> nx.MultiDiGraph:
    """
    Build a directed multigraph from triples: subject --relation--> object.
    MultiDiGraph (not DiGraph) because the same subject/object pair can
    legitimately have multiple different relations across different verses.
    """
    graph = nx.MultiDiGraph()
    for triple in triples:
        graph.add_node(triple.subject, node_type="entity")
        graph.add_node(triple.object, node_type="entity")
        graph.add_edge(
            triple.subject, triple.object,
            relation=triple.relation,
            source_verse_key=triple.source_verse_key,
        )
    return graph


def build_context_graph(context_items: List[dict]) -> nx.MultiDiGraph:
    """Build Graph_C from Phase 2's retrieved context list."""
    triples = extract_triples_from_context(context_items)
    return build_graph(triples)


def build_response_graph(response_text: str) -> nx.MultiDiGraph:
    """Build Graph_R from the LLM's candidate response text."""
    triples = extract_triples(response_text)
    return build_graph(triples)


def graph_summary(graph: nx.MultiDiGraph, name: str = "Graph") -> str:
    edges = list(graph.edges(data=True))
    lines = [f"{name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"]
    for u, v, data in edges[:10]:
        lines.append(f"  {u} --[{data.get('relation')}]--> {v}")
    if len(edges) > 10:
        lines.append(f"  ... and {len(edges) - 10} more")
    return "\n".join(lines)


# --- Self-test ---
if __name__ == "__main__":
    print("[TEST 1] Building a graph from a small triple set...")
    triples = extract_triples("خلق الله السماوات والأرض", verse_key="7:54")
    graph = build_graph(triples)
    print(graph_summary(graph, "Test graph"))
    assert graph.number_of_nodes() >= 2
    assert graph.number_of_edges() >= 1
    print("[PASS]\n")

    print("[TEST 2] Building Graph_C from a realistic context list...")
    fake_context = [
        {"verse_key": "1:1", "source_type": "verse", "text": "بسم الله الرحمن الرحيم"},
        {"verse_key": "21:83", "source_type": "tafsir", "text": "أيوب صبر على البلاء"},
        {"verse_key": "2:255", "source_type": "verse", "text": "الله لا اله الا هو الحي القيوم"},
    ]
    graph_c = build_context_graph(fake_context)
    print(graph_summary(graph_c, "Graph_C"))
    assert graph_c.number_of_edges() >= 3
    print("[PASS]\n")

    print("[TEST 3] Building Graph_R from a candidate LLM response...")
    response_text = "خلق الله الانسان من طين. وأمر الله الملائكة بالسجود لادم."
    graph_r = build_response_graph(response_text)
    print(graph_summary(graph_r, "Graph_R"))
    assert graph_r.number_of_edges() >= 2
    print("[PASS]\n")

    print("[RESULT] All E2 self-tests passed.")
