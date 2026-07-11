"""
A2 (FIXED) - Download real ARABIC Tafsir Ibn Kathir text, verse by verse.

Why this replaces the previous download_tafsir.py:
    - The old script pulled tafsir resource_id=169 from api.quran.com, which
      is the ENGLISH Ibn Kathir translation, not Arabic. Fine-tuning Arabic
      embeddings needs Arabic-register tafsir text.

Source used (free, no API key, verified working):
    spa5k/tafsir_api on GitHub - static JSON files, one per verse.
    https://github.com/spa5k/tafsir_api
    Arabic Ibn Kathir edition slug: "ar-tafsir-ibn-kathir"
    URL pattern: https://raw.githubusercontent.com/spa5k/tafsir_api/main/tafsir/{slug}/{surah}/{ayah}.json

Requirements:
    pip install requests
"""

import json
import os
import sys
import time
import argparse
import requests

RAW_DIR = os.path.join("quranNLP", "data", "raw") if os.path.exists("quranNLP") else "data/raw"
OUTPUT_PATH = os.path.join(RAW_DIR, "tafsir_ibn_kathir_arabic.json")

TAFSIR_SLUG = "ar-tafsir-ibn-kathir"
BASE_URL = f"https://raw.githubusercontent.com/spa5k/tafsir_api/main/tafsir/{TAFSIR_SLUG}"

# Verse counts per surah (1-indexed), standard Uthmani mushaf - needed because
# this API is queried ayah-by-ayah rather than returning a whole surah.
SURAH_AYAH_COUNTS = [
    7, 286, 200, 176, 120, 165, 206, 75, 129, 109, 123, 111, 43, 52, 99, 128,
    111, 110, 98, 135, 112, 78, 118, 64, 77, 227, 93, 88, 69, 60, 34, 30, 73,
    54, 45, 83, 182, 88, 75, 85, 54, 53, 89, 59, 37, 35, 38, 29, 18, 45, 60,
    49, 62, 55, 78, 96, 29, 22, 24, 13, 14, 11, 11, 18, 12, 12, 30, 52, 52,
    44, 28, 28, 20, 56, 40, 31, 50, 40, 46, 42, 29, 19, 36, 25, 22, 17, 19,
    26, 30, 20, 15, 21, 11, 8, 8, 19, 5, 8, 8, 11, 11, 8, 3, 9, 5, 4, 7, 3,
    6, 3, 5, 4, 5, 6,
]


def _ensure_output_dir():
    os.makedirs(RAW_DIR, exist_ok=True)


def fetch_verse_tafsir(surah: int, ayah: int) -> str:
    url = f"{BASE_URL}/{surah}/{ayah}.json"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json().get("text", "")


def fetch_all(limit_surahs: int, delay: float = 0.03) -> dict:
    results = {}
    failures = []
    total_verses = sum(SURAH_AYAH_COUNTS[:limit_surahs])
    done = 0

    for surah in range(1, limit_surahs + 1):
        ayah_count = SURAH_AYAH_COUNTS[surah - 1]
        for ayah in range(1, ayah_count + 1):
            verse_key = f"{surah}:{ayah}"
            try:
                text = fetch_verse_tafsir(surah, ayah)
                results[verse_key] = text
            except Exception as exc:  # noqa: BLE001
                failures.append((verse_key, str(exc)))
            done += 1
            if done % 200 == 0 or done == total_verses:
                print(f"  ... {done}/{total_verses} verses fetched "
                      f"({len(failures)} failures so far)")
            time.sleep(delay)

    print(f"\n[OK] Fetched real Arabic tafsir for {len(results)} verses, "
          f"{len(failures)} failures.")
    if failures[:5]:
        print("  Sample failures:", failures[:5])
    return results


def verify_no_placeholder_text(results: dict) -> bool:
    """
    Guard against ever silently shipping fake/placeholder data again: fail
    loudly if suspiciously many verses have IDENTICAL text (a real tafsir
    corpus should have near-zero exact duplicates across different verses,
    aside from very short, genuinely repeated phrases).
    """
    from collections import Counter
    counts = Counter(results.values())
    most_common_text, most_common_count = counts.most_common(1)[0] if counts else ("", 0)

    duplicate_ratio = most_common_count / max(len(results), 1)
    print(f"\nPlaceholder check: most repeated text appears {most_common_count} "
          f"times ({duplicate_ratio:.1%} of all verses).")
    print(f"  sample: {most_common_text[:80]!r}")

    passed = duplicate_ratio < 0.05  # a real corpus should not be >5% one repeated string
    print(f"[{'PASS' if passed else 'FAIL'}] No placeholder/duplicate-text pattern detected.")
    return passed


def save_output(results: dict):
    _ensure_output_dir()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"tafsir_edition": TAFSIR_SLUG, "language": "arabic", "verses": results},
                   f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {OUTPUT_PATH} ({len(results)} verses)")


def main():
    parser = argparse.ArgumentParser(description="A2 (fixed) - Download real Arabic Ibn Kathir tafsir.")
    parser.add_argument("--limit-surahs", type=int, default=3,
                         help="Fetch tafsir up to this surah number (default 3, fast test run). "
                              "Use 114 for the full corpus (6,236 requests - slower).")
    args = parser.parse_args()

    print(f"[STEP] Fetching real Arabic Ibn Kathir tafsir for surahs 1-{args.limit_surahs}...")
    results = fetch_all(args.limit_surahs)

    ok = verify_no_placeholder_text(results) and len(results) > 0
    save_output(results)

    print(f"\n[RESULT] A2 (fixed) tafsir acquisition {'PASSED' if ok else 'FAILED'} verification.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
