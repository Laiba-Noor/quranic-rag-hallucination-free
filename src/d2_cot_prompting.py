"""
D2 - Chain-of-Thought Prompt Templates.

Turns retrieved context (from D1's loop) into a structured prompt that
instructs an LLM to reason step by step, citing specific verse_key sources
for every claim - never an unexplained final answer.

The LLM call itself is dependency-injected (call_llm_fn), so this file has
NO hard dependency on any specific provider (Gemini, Anthropic, OpenAI,
etc.) and can be tested and used without an API key wired in yet.

A grounding check (validate_citations) is included so the response format
stays parseable for Phase 3's triple extraction / hallucination guardrail
later - loosely structured now, but consistently citable.
"""

import re
from dataclasses import dataclass
from typing import Callable, List, Dict, Optional


COT_SYSTEM_INSTRUCTIONS = """You are a careful assistant answering questions about the Quran using ONLY the provided context (verses and Tafsir passages). You must:

1. Respond ENTIRELY IN ARABIC. Do not write any English sentences, even for meta-commentary like "the context does not mention this" - write that in Arabic too (e.g. "لا يوجد ذكر لهذا في السياق المقدم").
2. Reason step by step, building your answer from the provided context only.
3. For every claim, cite the exact verse_key it comes from, using the format [verse_key].
4. If the provided context does not fully answer the question, write exactly the single line "غير كاف" (insufficient) as your entire Answer section - do not write an explanation of what is missing.
5. Never state a claim without a citation to the context provided below.

Structure your answer as:
Reasoning:
<step-by-step reasoning, each step ending with a citation like [2:255]>

Answer:
<final answer, itself citing sources>
"""


def format_context_block(context: List[Dict], max_chars_per_item: int = 400) -> str:
    """
    Turn D1's retrieved context list into a numbered, citable text block.

    max_chars_per_item caps each passage's length before it goes into the
    prompt. This matters in practice: tafsir passages can be long, and with
    top_k=5 results an untruncated context block can push a single request
    over a provider's tokens-per-minute limit (this is what caused the 413
    rate-limit error - not a bug in the loop itself, just an oversized
    prompt). Truncating here fixes the root cause rather than only reacting
    to the error after the fact.
    """
    lines = []
    for item in context:
        tag = item.get("source_type", "unknown")
        vk = item.get("verse_key", "?")
        text = item.get("text", "")
        if max_chars_per_item and len(text) > max_chars_per_item:
            text = text[:max_chars_per_item].rstrip() + "\u2026"
        lines.append(f"[{vk}] ({tag}): {text}")
    return "\n".join(lines)


def build_cot_prompt(query: str, context: List[Dict], max_chars_per_item: int = 400) -> str:
    """
    Build the full Chain-of-Thought prompt: system instructions + retrieved
    context + the user's question.
    """
    context_block = format_context_block(context, max_chars_per_item=max_chars_per_item)
    prompt = (
        f"{COT_SYSTEM_INSTRUCTIONS}\n"
        f"---\nContext:\n{context_block}\n---\n\n"
        f"Question: {query}\n"
    )
    return prompt


@dataclass
class CoTResult:
    query: str
    raw_response: str
    reasoning: str
    answer: str
    cited_verse_keys: List[str]
    grounding_ok: bool  # True if every citation actually appears in the given context


def extract_citations(text: str) -> List[str]:
    """Extract verse_key citations like [2:255] from a response, deduplicated
    but preserving first-seen order."""
    found = re.findall(r"\[(\d+:\d+)\]", text)
    seen = []
    for key in found:
        if key not in seen:
            seen.append(key)
    return seen


def split_reasoning_and_answer(response_text: str) -> (str, str):
    """Split a response into its Reasoning: and Answer: sections."""
    reasoning_match = re.search(r"Reasoning:\s*(.*?)(?=Answer:|$)", response_text, re.DOTALL)
    answer_match = re.search(r"Answer:\s*(.*)", response_text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    answer = answer_match.group(1).strip() if answer_match else response_text.strip()
    return reasoning, answer


def validate_citations(cited_keys: List[str], context: List[Dict]) -> bool:
    """
    Grounding check: every cited verse_key must actually be present in the
    context that was given to the model. This is a cheap, immediate sanity
    check - Phase 3's full triple-extraction guardrail does the deeper,
    content-level version of this same idea.
    """
    context_keys = {item.get("verse_key") for item in context}
    if not cited_keys:
        return False  # no citations at all is not acceptable output
    return all(key in context_keys for key in cited_keys)


def run_cot_reasoning(
    query: str,
    context: List[Dict],
    call_llm_fn: Callable[[str], str],
    max_chars_per_item: int = 400,
) -> CoTResult:
    """
    Build the prompt, call the injected LLM function, parse and validate
    the response. call_llm_fn takes a prompt string and returns the model's
    raw text response - swap in any real provider without touching this logic.
    """
    prompt = build_cot_prompt(query, context, max_chars_per_item=max_chars_per_item)
    raw_response = call_llm_fn(prompt)

    reasoning, answer = split_reasoning_and_answer(raw_response)
    cited_keys = extract_citations(raw_response)
    grounded = validate_citations(cited_keys, context)

    return CoTResult(
        query=query,
        raw_response=raw_response,
        reasoning=reasoning,
        answer=answer,
        cited_verse_keys=cited_keys,
        grounding_ok=grounded,
    )


def print_cot_result(result: CoTResult):
    print(f"\n{'='*60}")
    print(f"Query: {result.query}")
    print(f"\nReasoning:\n{result.reasoning}")
    print(f"\nAnswer:\n{result.answer}")
    print(f"\nCited verse keys: {result.cited_verse_keys}")
    print(f"Grounding check: {'PASS' if result.grounding_ok else 'FAIL'} "
          f"(every citation must appear in the given context)")
    print("=" * 60)


# --- Self-test with a fake LLM function, no API key needed ---
if __name__ == "__main__":
    fake_context = [
        {"verse_key": "1:1", "source_type": "verse", "text": "In the name of Allah, the Most Merciful"},
        {"verse_key": "1:2", "source_type": "tafsir", "text": "This verse establishes Allah's mercy as foundational"},
    ]

    def fake_llm_call(prompt: str) -> str:
        return (
            "Reasoning:\n"
            "The opening verse establishes divine mercy as a core theme [1:1]. "
            "The accompanying commentary reinforces that this mercy is foundational "
            "to the text as a whole [1:2].\n\n"
            "Answer:\n"
            "Mercy is presented as one of the Quran's foundational themes, "
            "established from the very first verse [1:1] and reinforced by "
            "classical commentary [1:2]."
        )

    print("[TEST] Running run_cot_reasoning with a fake LLM function...")
    result = run_cot_reasoning("What does the Quran say about mercy?", fake_context, fake_llm_call)
    print_cot_result(result)

    assert result.grounding_ok, "Expected all citations to be grounded in context"
    assert set(result.cited_verse_keys) == {"1:1", "1:2"}, "Expected both context keys cited"
    print("\n[PASS] D2 self-test passed: citations extracted and grounding validated correctly.")

    def bad_llm_call(prompt: str) -> str:
        return "Reasoning:\nSome claim [9:99].\n\nAnswer:\nUngrounded answer [9:99]."

    bad_result = run_cot_reasoning("test", fake_context, bad_llm_call)
    assert not bad_result.grounding_ok, "Expected grounding check to FAIL for an uncited-in-context key"
    print("[PASS] D2 self-test passed: ungrounded citation correctly detected as FAIL.")
