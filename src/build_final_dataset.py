"""
build_final_dataset.py - Replaces the broken export_sync_point1.py and
cross_reference.py. Joins:
    - quranNLP/shared/data/clean_verses.csv          (from A3 fixed)
    - quranNLP/data/raw/tafsir_ibn_kathir_arabic.json (from A2 fixed)

into the final Sync Point 1 handoff file Member B needs:
    quranNLP/shared/data/final_cross_reference_index.csv
    columns: verse_key, related_verse_keys, tafsir_passage

Design notes:
    - Not every verse has its own individual tafsir block (classical
      commentary often covers several verses in one passage under the first
      verse's key). For a verse with no direct entry, this script falls back
      to the NEAREST PRECEDING verse's real tafsir within the same surah,
      and marks it explicitly in has_direct_tafsir so nobody downstream
      mistakes a fallback for a direct commentary on that exact verse.
    - HARD GUARD: refuses to write the output file if more than 5% of rows
      end up with identical tafsir text - this is exactly the bug that
      shipped placeholder data before, and this check makes it impossible
      to silently happen again.
"""

import os
import csv
import json
import sys
from collections import Counter

BASE_DIR = "quranNLP" if os.path.exists("quranNLP") else "."
SHARED_DIR = os.path.join(BASE_DIR, "shared", "data")
CLEAN_VERSES_PATH = os.path.join(SHARED_DIR, "clean_verses.csv")
TAFSIR_PATH = os.path.join(BASE_DIR, "data", "raw", "tafsir_ibn_kathir_arabic.json")
OUTPUT_PATH = os.path.join(SHARED_DIR, "final_cross_reference_index.csv")

MAX_DUPLICATE_RATIO = 0.05  # fail the build if >5% of rows share identical tafsir text


def load_clean_verses() -> dict:
    if not os.path.exists(CLEAN_VERSES_PATH):
        print(f"[FAIL] {CLEAN_VERSES_PATH} not found. Run a3_preprocess_pipeline_fixed.py first.",
              file=sys.stderr)
        sys.exit(1)
    with open(CLEAN_VERSES_PATH, encoding="utf-8") as f:
        return {row["verse_key"]: row for row in csv.DictReader(f)}


def load_tafsir() -> dict:
    if not os.path.exists(TAFSIR_PATH):
        print(f"[FAIL] {TAFSIR_PATH} not found. Run a2_tafsir_acquisition_arabic.py first.",
              file=sys.stderr)
        sys.exit(1)
    with open(TAFSIR_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("verses", {})


def fill_tafsir_with_nearest_preceding(verse_keys: list, tafsir: dict) -> dict:
    """
    For verses with no direct tafsir entry, fall back to the nearest
    preceding verse's tafsir WITHIN THE SAME SURAH. Returns
    verse_key -> (text, has_direct_tafsir: bool).
    """
    filled = {}
    last_text_by_surah = {}

    for vk in verse_keys:
        surah = vk.split(":")[0]
        if vk in tafsir and tafsir[vk].strip():
            filled[vk] = (tafsir[vk], True)
            last_text_by_surah[surah] = tafsir[vk]
        else:
            fallback_text = last_text_by_surah.get(surah, "")
            filled[vk] = (fallback_text, False)

    return filled


def build_cross_references(verse_keys: list) -> dict:
    """Simple adjacent-verse cross-reference within the same surah."""
    refs = {}
    for vk in verse_keys:
        surah, ayah = vk.split(":")
        next_key = f"{surah}:{int(ayah) + 1}"
        refs[vk] = [next_key] if next_key in verse_keys else []
    return refs


def verify_no_placeholder(rows: list) -> bool:
    texts = [r["tafsir_passage"] for r in rows if r["tafsir_passage"]]
    if not texts:
        print("[FAIL] No tafsir text present at all.")
        return False

    counts = Counter(texts)
    most_common_text, most_common_count = counts.most_common(1)[0]
    ratio = most_common_count / len(texts)

    direct_count = sum(1 for r in rows if r["has_direct_tafsir"] == "True")
    direct_ratio = direct_count / len(rows)

    print("\nValidation:")
    print(f"  most repeated tafsir text: {most_common_count}/{len(texts)} rows "
          f"({ratio:.1%}) -> {'FAIL' if ratio > MAX_DUPLICATE_RATIO else 'OK'}")
    print(f"  verses with DIRECT tafsir (not fallback): {direct_count}/{len(rows)} "
          f"({direct_ratio:.1%})")
    print(f"  sample repeated text: {most_common_text[:80]!r}")

    passed = ratio <= MAX_DUPLICATE_RATIO
    print(f"\n[{'PASS' if passed else 'FAIL'}] Placeholder/duplicate-text guard.")
    return passed


def main():
    clean_verses = load_clean_verses()
    tafsir = load_tafsir()
    verse_keys = sorted(clean_verses.keys(), key=lambda k: (int(k.split(":")[0]), int(k.split(":")[1])))

    print(f"[OK] Loaded {len(clean_verses)} clean verses, {len(tafsir)} tafsir entries.")

    tafsir_filled = fill_tafsir_with_nearest_preceding(verse_keys, tafsir)
    cross_refs = build_cross_references(verse_keys)

    rows = []
    for vk in verse_keys:
        text, has_direct = tafsir_filled[vk]
        rows.append({
            "verse_key": vk,
            "clean_verse": clean_verses[vk]["clean_verse"],
            "related_verse_keys": ";".join(cross_refs[vk]),
            "tafsir_passage": text,
            "has_direct_tafsir": str(has_direct),
        })

    ok = verify_no_placeholder(rows)

    if not ok:
        print("\n[ABORTED] Refusing to write output file - placeholder/duplicate-text "
              "pattern detected. Run a2_tafsir_acquisition_arabic.py with a larger "
              "--limit-surahs (or --limit-surahs 114 for the full corpus) and retry.",
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(SHARED_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "verse_key", "clean_verse", "related_verse_keys",
            "tafsir_passage", "has_direct_tafsir",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[SAVED] {OUTPUT_PATH} ({len(rows)} rows)")
    print("[RESULT] Sync Point 1 handoff file ready for Member B.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
