"""
A3 (FIXED) - Clean Arabic verse text ONLY. This replaces the broken
preprocess_pipeline.py, which incorrectly hardcoded placeholder tafsir text
into what should have been a pure text-cleaning step.

Tafsir joining is handled separately by build_final_dataset.py (below), which
merges this file with the REAL tafsir from a2_tafsir_acquisition_arabic.py.
Keeping these as two separate steps prevents the original bug (fake tafsir
silently baked into the "cleaned" file) from happening again.

Input: quranNLP/data/raw/quran-uthmani-min.xml (Roma's existing download)
Output: quranNLP/shared/data/clean_verses.csv
"""

import os
import re
import csv
import sys
import xml.etree.ElementTree as ET

BASE_DIR = "quranNLP" if os.path.exists("quranNLP") else "."
XML_PATH = os.path.join(BASE_DIR, "data", "raw", "quran-uthmani-min.xml")
OUTPUT_DIR = os.path.join(BASE_DIR, "shared", "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "clean_verses.csv")

DIACRITICS = re.compile(r"[\u064B-\u0652]")
ALEF_VARIANTS = re.compile(r"[\u0622\u0623\u0625]")


def clean_arabic_text(text: str) -> str:
    if not text:
        return ""
    text = DIACRITICS.sub("", text)          # strip tashkeel
    text = ALEF_VARIANTS.sub("\u0627", text)  # أ إ آ -> ا
    return text.strip()


def parse_tanzil_xml(xml_path: str) -> dict:
    """Parse the Tanzil XML into verse_key -> clean text. Returns {} if missing."""
    if not os.path.exists(xml_path):
        print(f"[FAIL] XML file not found at {xml_path}", file=sys.stderr)
        return {}

    print(f"[STEP] Parsing Tanzil XML: {xml_path}")
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Could not parse XML: {exc}", file=sys.stderr)
        return {}

    verses = {}
    for sura in root.findall(".//sura"):
        sura_index = sura.get("index")
        for aya in sura.findall(".//aya"):
            aya_index = aya.get("index")
            raw_text = aya.get("text")
            if raw_text:
                verse_key = f"{sura_index}:{aya_index}"
                verses[verse_key] = {
                    "verse_key": verse_key,
                    "text_original": raw_text,
                    "clean_verse": clean_arabic_text(raw_text),
                }
    return verses


def verify(verses: dict) -> bool:
    ok_count = len(verses) == 6236
    sample = verses.get("2:255")
    print(f"\nVerification: {len(verses)} verses parsed (expected 6236) "
          f"-> {'OK' if ok_count else 'MISMATCH'}")
    if sample:
        print(f"  2:255 original : {sample['text_original'][:60]}...")
        print(f"  2:255 cleaned  : {sample['clean_verse'][:60]}...")
    return ok_count and sample is not None


def save_output(verses: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["verse_key", "text_original", "clean_verse"])
        writer.writeheader()
        writer.writerows(verses.values())
    print(f"[SAVED] {OUTPUT_PATH} ({len(verses)} verses)")


def main():
    verses = parse_tanzil_xml(XML_PATH)
    if not verses:
        sys.exit(1)
    ok = verify(verses)
    save_output(verses)
    print(f"\n[RESULT] A3 (fixed) preprocessing {'PASSED' if ok else 'completed with warnings'}.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
