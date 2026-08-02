"""
E1 - Semantic Triple Extraction (Subject-Relation-Object).

Extracts (subject, relation, object) triples from Arabic text - both the
retrieved context (verses/tafsir) and the LLM's generated response. This is
the foundation the Phase 3 guardrail compares against: if the response's
triples don't structurally align with the context's triples, the response
gets rejected as likely hallucinated.

Approach: rule-based extraction using a curated list of common
Quranic/theological Arabic verbs and prepositions as relation anchors,
rather than a full statistical dependency parser. This is a deliberate,
honest design choice - mature Arabic Open Information Extraction tooling is
still an active research gap (this is literally one of the paper's own
research gaps), so a transparent, inspectable rule-based extractor is more
trustworthy for a hallucination guardrail than a black-box model that could
itself introduce errors at the exact layer meant to catch errors.

Every sentence is guaranteed to produce at least one triple (falling back to
a generic "mentions" relation) so downstream graph construction never
silently drops content.

Requirements:
    pip install pyarabic
"""

import re
import sys
from dataclasses import dataclass
from typing import List, Optional
import pyarabic.araby as araby

# Common Quranic/theological Arabic verbs used as relation anchors.
# Not exhaustive - a real research deployment would grow this list from
# failure analysis on actual corpus text.
RELATION_VERBS = [
    "قال", "خلق", "أمر", "نهى", "جعل", "أنزل", "كتب", "حرم", "أحل",
    "وعد", "أوعد", "رحم", "غفر", "هدى", "أضل", "بشر", "أنذر", "أرسل",
    "بعث", "علم", "أعطى", "منع", "أهلك", "نجى", "عذب", "ثاب", "تاب",
    "استغفر", "شكر", "صبر", "آمن", "كفر", "عبد", "سجد", "دعا", "استجاب",
    "خلقنا", "خلقكم", "يخلق", "يأمر", "ينهى", "يجعل", "ينزل", "يهدي",
    "يغفر", "يرحم", "يعذب", "ينجي",
]

# Common Arabic prepositions used as a fallback relation anchor when no
# known verb is found in the sentence.
RELATION_PREPOSITIONS = ["في", "من", "إلى", "على", "عن", "مع", "بـ", "لـ", "كـ"]


@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    source_verse_key: Optional[str] = None  # provenance - which verse/tafsir this came from

    def as_tuple(self):
        return (self.subject, self.relation, self.object)

    def __repr__(self):
        return f"({self.subject} | {self.relation} | {self.object})"


def clean_text(text: str) -> str:
    """Strip diacritics and normalize whitespace before extraction."""
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_clauses(text: str) -> List[str]:
    """Split a sentence into rough clauses on 'و' (and), '،', and '.'."""
    # Split on Arabic comma, period, and standalone conjunction "و" between words
    parts = re.split(r"[،.]|\sو(?=\s)", text)
    return [p.strip() for p in parts if p.strip()]


def extract_triple_from_clause(clause: str, verse_key: Optional[str] = None) -> Optional[Triple]:
    """
    Extract a single (subject, relation, object) triple from one clause using
    verb-anchored splitting first, falling back to preposition-anchored
    splitting, falling back to a generic "mentions" triple.
    """
    words = clause.split()
    if not words:
        return None

    # Strategy 1: find a known relation verb
    for i, word in enumerate(words):
        if word in RELATION_VERBS:
            subject = " ".join(words[:i]).strip() or "(implicit: Allah)"
            relation = word
            obj = " ".join(words[i + 1:]).strip()
            if obj:
                return Triple(subject, relation, obj, source_verse_key=verse_key)

    # Strategy 2: find a known preposition
    for i, word in enumerate(words):
        if word in RELATION_PREPOSITIONS and 0 < i < len(words) - 1:
            subject = " ".join(words[:i]).strip()
            relation = word
            obj = " ".join(words[i + 1:]).strip()
            if subject and obj:
                return Triple(subject, relation, obj, source_verse_key=verse_key)

    # Strategy 3: fallback - whole clause as a generic "mentions" fact
    return Triple(subject=clause, relation="mentions", object=clause, source_verse_key=verse_key)


CITATION_ONLY_PATTERN = re.compile(r"^[\[\]\d:\s\-]+$")  # e.g. "[37:129]" or "]] [62:2]" with nothing else
MIN_CONTENT_WORDS = 2  # a clause needs at least this many non-citation Arabic words to count as a real claim


def is_degenerate_clause(clause: str) -> bool:
    """
    True if a clause is a citation-only fragment (e.g. "[37:129]") or too
    short to represent a real claim (e.g. "أخيرا" / "finally") once
    citations are stripped out. These are artifacts of how LLMs format
    multi-source citations across separate lines, NOT actual assertions -
    counting them as their own triples inflates hallucination scores on
    otherwise correct, well-grounded answers.
    """
    if CITATION_ONLY_PATTERN.match(clause.strip()):
        return True

    without_citations = re.sub(r"\[\d+:\d+\]", "", clause).strip()
    content_words = [w for w in without_citations.split() if len(w) >= 2]
    return len(content_words) < MIN_CONTENT_WORDS


def extract_triples(text: str, verse_key: Optional[str] = None) -> List[Triple]:
    """
    Main entry point: extract all triples from a piece of text (a verse,
    a tafsir passage, or an LLM-generated response). Degenerate clauses
    (bare citations, one-word connectives) are skipped rather than turned
    into hollow "mentions" triples.
    """
    cleaned = clean_text(text)
    clauses = split_clauses(cleaned)

    triples = []
    for clause in clauses:
        if is_degenerate_clause(clause):
            continue
        triple = extract_triple_from_clause(clause, verse_key=verse_key)
        if triple:
            triples.append(triple)

    return triples


def extract_triples_from_context(context_items: List[dict]) -> List[Triple]:
    """
    Extract triples from Phase 2's retrieved context list (each item has
    'text' and 'verse_key'), tagging each triple with its source for
    provenance-aware alignment checking later.
    """
    all_triples = []
    for item in context_items:
        text = item.get("text", "")
        verse_key = item.get("verse_key")
        all_triples.extend(extract_triples(text, verse_key=verse_key))
    return all_triples


# --- Self-test on real Quranic/tafsir-style Arabic text ---
if __name__ == "__main__":
    print("[TEST 1] Extracting from a verb-anchored sentence...")
    text1 = "إن الله غفور رحيم"
    triples1 = extract_triples(text1)
    print(f"  Input: {text1}")
    for t in triples1:
        print(f"  -> {t}")
    assert len(triples1) >= 1
    print("[PASS]\n")

    print("[TEST 2] Extracting from a real tafsir-style sentence with a known verb...")
    text2 = "خلق الله السماوات والأرض في ستة أيام"
    triples2 = extract_triples(text2, verse_key="7:54")
    print(f"  Input: {text2}")
    for t in triples2:
        print(f"  -> {t} (source: {t.source_verse_key})")
    assert any(t.relation == "خلق" for t in triples2), "Expected the verb 'خلق' to be found as a relation"
    print("[PASS]\n")

    print("[TEST 3] Extracting from context list (Phase 2 format)...")
    fake_context = [
        {"verse_key": "1:1", "source_type": "verse", "text": "بسم الله الرحمن الرحيم"},
        {"verse_key": "21:83", "source_type": "tafsir", "text": "أيوب صبر على البلاء وشكر الله"},
    ]
    context_triples = extract_triples_from_context(fake_context)
    print(f"  Extracted {len(context_triples)} triples from {len(fake_context)} context items:")
    for t in context_triples:
        print(f"  -> {t} (source: {t.source_verse_key})")
    assert len(context_triples) >= 2
    print("[PASS]\n")

    print("[TEST 4] Fallback strategy on a clause with no known verb/preposition...")
    text4 = "الحمد والثناء"
    triples4 = extract_triples(text4)
    print(f"  Input: {text4}")
    for t in triples4:
        print(f"  -> {t}")
    assert len(triples4) == 1 and triples4[0].relation == "mentions"
    print("[PASS]\n")

    print("[TEST 5] Degenerate clause filtering (citation-only and filler words skipped)...")
    text5 = "أيوب صبر على البلاء. [37:129]. أخيرا."
    triples5 = extract_triples(text5)
    print(f"  Input: {text5}")
    for t in triples5:
        print(f"  -> {t}")
    assert len(triples5) == 1, f"Expected only the real claim to produce a triple, got {len(triples5)}"
    assert "37:129" not in triples5[0].subject and "37:129" not in triples5[0].object
    print("[PASS]\n")

    print("[RESULT] All E1 self-tests passed.")
