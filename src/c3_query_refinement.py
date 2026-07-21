"""
C3 - Query-Refinement Logic.

Builds the exact function Laiba's D1 (AgenticRetriever) and D3
(TheologicalAgent) expect:
    refine_fn(query: str, results: List[Dict]) -> str

When the agent's first retrieval round is judged insufficient, this
produces a reformulated query for the next round, using three real,
rule-based strategies (no LLM call needed, so this works standalone and
fast inside the loop):

    1. Synonym/related-term expansion using a curated Arabic theological
       term dictionary (mercy -> mercy + forgiveness + compassion, etc.)
    2. Broadening: strip overly specific qualifiers that may be narrowing
       the search too much (numbers, rare proper nouns).
    3. Steering: pull salient Arabic content words from the top (even if
       insufficient) retrieved result's text and fold them into the next
       query, nudging the search toward the neighborhood it already found.

Strategies are tried in order; the first one that actually changes the
query is used, so the loop always makes forward progress rather than
looping on an identical query.
"""

import re
import sys
from typing import List, Dict, Optional

# A small, curated Classical Arabic theological synonym/related-term map.
# Not exhaustive - meant to be extended over time as real failure cases
# surface during testing.
THEOLOGICAL_SYNONYMS = {
    "رحمة": ["مغفرة", "رأفة", "عفو"],           # mercy -> forgiveness, compassion, pardon
    "صبر": ["ثبات", "احتساب", "تحمل"],           # patience -> steadfastness, endurance
    "ايمان": ["يقين", "تصديق"],                  # faith -> certainty, belief
    "عدل": ["قسط", "انصاف"],                      # justice -> equity, fairness
    "توبة": ["استغفار", "انابة"],                 # repentance -> seeking forgiveness, returning to God
    "شكر": ["حمد", "امتنان"],                     # gratitude -> praise, thankfulness
    "خوف": ["تقوى", "خشية"],                      # fear -> God-consciousness, awe
}

# Common overly-narrow qualifiers that can be stripped to broaden a search
# when the specific/narrow version returned nothing useful.
NARROWING_PATTERNS = [
    r"\bفي سورة \S+\b",   # "in Surah X" - drop the specific surah constraint
    r"\bالآية \d+\b",       # "verse N" - drop a specific verse number reference
]


def synonym_expansion(query: str) -> Optional[str]:
    """Strategy 1: append related theological terms for any matched keyword."""
    for term, synonyms in THEOLOGICAL_SYNONYMS.items():
        if term in query:
            expansion = " ".join(synonyms)
            return f"{query} {expansion}"
    return None


def broaden_query(query: str) -> Optional[str]:
    """Strategy 2: strip overly narrow qualifiers to widen the search."""
    broadened = query
    changed = False
    for pattern in NARROWING_PATTERNS:
        new_query = re.sub(pattern, "", broadened).strip()
        if new_query != broadened:
            broadened = new_query
            changed = True
    return broadened if changed else None


def extract_content_words(text: str, max_words: int = 4) -> List[str]:
    """Pull a handful of longer Arabic words from text as likely content-bearing terms."""
    words = re.findall(r"[\u0600-\u06FF]+", text)
    # Prefer longer words (short words are usually function words: articles, prepositions)
    content_words = sorted(set(w for w in words if len(w) >= 4), key=len, reverse=True)
    return content_words[:max_words]


def steer_toward_results(query: str, results: List[Dict]) -> Optional[str]:
    """Strategy 3: fold salient terms from the (weak) top result into the next query."""
    if not results:
        return None
    top_text = results[0].get("text", "")
    steering_terms = extract_content_words(top_text)
    if not steering_terms:
        return None
    return f"{query} {' '.join(steering_terms)}"


def refine_query(query: str, results: List[Dict]) -> str:
    """
    The refine_fn Laiba's agent loop calls. Tries strategies in order and
    returns the first one that actually changes the query. Falls back to
    the original query unchanged only if none of the strategies apply -
    logged clearly so this is visible, not silent.
    """
    for strategy_name, strategy_fn in [
        ("synonym_expansion", synonym_expansion),
        ("broaden_query", broaden_query),
        ("steer_toward_results", lambda q: steer_toward_results(q, results)),
    ]:
        refined = strategy_fn(query)
        if refined and refined != query:
            print(f"[C3] Refinement strategy used: {strategy_name}")
            return refined

    print("[C3][WARN] No refinement strategy produced a change - "
          "returning original query unchanged.")
    return query


# --- Self-test, no external dependencies needed ---
if __name__ == "__main__":
    print("[TEST 1] Synonym expansion...")
    result = refine_query("ما فوائد الرحمة", [])
    print(f"  Input : ما فوائد الرحمة")
    print(f"  Output: {result}")
    assert result != "ما فوائد الرحمة", "Expected synonym expansion to change the query"
    assert "مغفرة" in result
    print("[PASS]\n")

    print("[TEST 2] Broadening an overly narrow query...")
    result2 = refine_query("الاستقامة في سورة البقرة الآية 5", [])
    print(f"  Input : الاستقامة في سورة البقرة الآية 5")
    print(f"  Output: {result2}")
    assert "في سورة" not in result2 and "الآية" not in result2
    print("[PASS]\n")

    print("[TEST 3] Steering toward weak retrieved result content...")
    fake_results = [{"verse_key": "2:100", "text": "الاستقامة والثبات على الطريق المستقيم"}]
    result3 = refine_query("سؤال غامض", fake_results)
    print(f"  Input : سؤال غامض")
    print(f"  Output: {result3}")
    assert result3 != "سؤال غامض"
    print("[PASS]\n")

    print("[TEST 4] No strategy applies - query returned unchanged with a warning...")
    result4 = refine_query("xyz", [])
    assert result4 == "xyz"
    print("[PASS]\n")

    print("[RESULT] All C3 self-tests passed.")
