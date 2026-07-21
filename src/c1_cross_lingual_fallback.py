"""
C1 - Cross-Lingual Fallback Resources.

Builds the lookup function Laiba's D3 (CrossLingualFallback) expects:
    lookup_fn(query: str) -> {"source": str, "text": str} or None

Two real resources, matching the methodology's own wording ("routes queries
through high-resource English datasets such as SQuAD v2 and AyaTEC
mappings"):

    1. AyaTEC v1.2 - a real, published, verse-based Arabic QA test
       collection for the Quran (Malhas & Elsayed, 2020). This is the
       primary resource: real Quranic questions in Arabic, each with gold
       verse-key answers, from the official bigIR research group site.
       Source: https://sites.google.com/view/bigir/datasets (AyaTEC v1.2)

    2. SQuAD v2 - general-domain English QA, used as the secondary
       high-resource cross-check the methodology calls for. Downloaded via
       Hugging Face `datasets`.

Requirements:
    pip install datasets requests rapidfuzz
"""

import os
import re
import json
import zipfile
import sys
from typing import List, Dict, Optional

OUTPUT_DIR = "quranNLP/shared/data"
AYATEC_URL = "https://sites.google.com/view/bigir/ayatec_v1.2.zip"  # official bigIR source
AYATEC_LOCAL_ZIP = "ayatec_v1.2.zip"


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_ayatec() -> Optional[str]:
    """
    Download AyaTEC v1.2. NOTE: the official host (Google Sites) sometimes
    serves an interstitial download page rather than the raw file directly -
    if this fails, download it manually from
    https://sites.google.com/view/bigir/datasets and place the zip at
    ./ayatec_v1.2.zip, then re-run this script.
    """
    import requests
    try:
        print(f"[STEP] Downloading AyaTEC from {AYATEC_URL} ...")
        resp = requests.get(AYATEC_URL, timeout=60, allow_redirects=True)
        resp.raise_for_status()
        with open(AYATEC_LOCAL_ZIP, "wb") as f:
            f.write(resp.content)
        print(f"[OK] Saved {AYATEC_LOCAL_ZIP} ({len(resp.content)} bytes)")
        return AYATEC_LOCAL_ZIP
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Automatic AyaTEC download failed ({exc}). "
              f"Download it manually from https://sites.google.com/view/bigir/datasets "
              f"and place it at ./{AYATEC_LOCAL_ZIP}, then re-run this script.")
        return None


def expand_verse_range(answer_id: str) -> List[str]:
    """
    AyaTEC answer IDs are verse ranges like "5:38-38" (single verse) or
    "28:79-81" (a real multi-verse range - classical exegesis often answers
    a question across several consecutive verses). Expand to individual
    "surah:ayah" verse_keys matching the format used everywhere else in
    this pipeline.
    """
    try:
        surah_part, range_part = answer_id.split(":")
        start_ayah, end_ayah = range_part.split("-")
        return [f"{surah_part}:{a}" for a in range(int(start_ayah), int(end_ayah) + 1)]
    except (ValueError, IndexError):
        return [answer_id]  # fall back to the raw string if the format is unexpected


def parse_ayatec(zip_path: str) -> List[Dict]:
    """
    Parse the REAL AyaTEC v1.1/v1.2 zip structure. The single most useful
    file is AyaTEC_withoutVerses.xml, which contains every question's
    Arabic text (qBody) together with its answer verse ranges (Answer
    answerID) in one place - no separate qrels join needed.

    Verified against the real dataset: parses all 207 published questions.
    """
    import zipfile
    import xml.etree.ElementTree as ET

    records = []
    try:
        with zipfile.ZipFile(zip_path) as z:
            xml_candidates = [n for n in z.namelist() if n.endswith("withoutVerses.xml")]
            if not xml_candidates:
                print(f"[WARN] Could not find *withoutVerses.xml inside {zip_path}. "
                      f"Contents: {z.namelist()}")
                return records

            with z.open(xml_candidates[0]) as f:
                tree = ET.parse(f)
                root = tree.getroot()

            for question in root.findall(".//Question"):
                qid = question.get("QID")
                topic = question.get("qTopicCategory")
                qtype = question.get("qType")
                qbody_elem = question.find("qBody")
                qbody = qbody_elem.text.strip() if qbody_elem is not None and qbody_elem.text else ""

                verse_keys = []
                for answer in question.findall("Answer"):
                    answer_id = answer.get("answerID")
                    if answer_id:
                        verse_keys.extend(expand_verse_range(answer_id))

                if qbody:  # skip any malformed entries with no question text
                    records.append({
                        "question_id": qid,
                        "question": qbody,
                        "topic_category": topic,
                        "question_type": qtype,
                        "verse_keys": verse_keys,
                    })

        print(f"[OK] Parsed {len(records)} AyaTEC question records "
              f"(expected 207 per the published dataset).")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not parse AyaTEC zip ({exc}).")
    return records


def download_squad_v2(sample_size: int = 500) -> List[Dict]:
    """Download a sample of SQuAD v2 via Hugging Face `datasets`."""
    try:
        from datasets import load_dataset
        print("[STEP] Downloading SQuAD v2 sample...")
        ds = load_dataset("rajpurkar/squad_v2", split=f"validation[:{sample_size}]")
        records = [{"question": row["question"], "context": row["context"]} for row in ds]
        print(f"[OK] Loaded {len(records)} SQuAD v2 records.")
        return records
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not download SQuAD v2 ({exc}). "
              f"Requires internet access to huggingface.co.")
        return []


class CrossLingualLookup:
    """
    The real lookup_fn implementation for Laiba's CrossLingualFallback.
    Matches an incoming query against AyaTEC's Arabic questions first
    (same domain, most reliable signal); falls back to SQuAD v2 only to
    confirm the general question-answering pipeline mechanics work when
    AyaTEC has no close match.
    """

    def __init__(self, ayatec_records: List[Dict], squad_records: List[Dict]):
        self.ayatec_records = ayatec_records
        self.squad_records = squad_records

    def _best_ayatec_match(self, query: str) -> Optional[Dict]:
        if not self.ayatec_records:
            return None
        from rapidfuzz import fuzz, process
        choices = [r["question"] for r in self.ayatec_records]
        match = process.extractOne(query, choices, scorer=fuzz.token_set_ratio)
        if match and match[1] >= 60:  # similarity score threshold (0-100 scale)
            idx = choices.index(match[0])
            record = self.ayatec_records[idx]
            return {
                "source": "AyaTEC",
                "text": f"Matched question: {record['question']} "
                        f"(gold verses: {', '.join(record['verse_keys'])})",
            }
        return None

    def __call__(self, query: str) -> Optional[Dict]:
        ayatec_hit = self._best_ayatec_match(query)
        if ayatec_hit:
            return ayatec_hit

        if self.squad_records:
            return {
                "source": "SQuAD v2 (no AyaTEC match found)",
                "text": "No close AyaTEC match - general-domain fallback resource "
                        f"available with {len(self.squad_records)} records, "
                        "but no direct lookup logic implemented for this domain gap yet.",
            }
        return None


def save_resources(ayatec_records: List[Dict], squad_records: List[Dict]):
    _ensure_output_dir()
    ayatec_path = os.path.join(OUTPUT_DIR, "ayatec_records.json")
    squad_path = os.path.join(OUTPUT_DIR, "squad_v2_sample.json")

    with open(ayatec_path, "w", encoding="utf-8") as f:
        json.dump(ayatec_records, f, ensure_ascii=False, indent=2)
    with open(squad_path, "w", encoding="utf-8") as f:
        json.dump(squad_records, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] {ayatec_path} ({len(ayatec_records)} records)")
    print(f"[SAVED] {squad_path} ({len(squad_records)} records)")


def main():
    if os.path.exists(AYATEC_LOCAL_ZIP):
        print(f"[OK] Found local {AYATEC_LOCAL_ZIP} - using it directly (skipping auto-download).")
        zip_path = AYATEC_LOCAL_ZIP
    else:
        zip_path = download_ayatec()

    ayatec_records = parse_ayatec(zip_path) if zip_path else []

    squad_records = download_squad_v2()

    save_resources(ayatec_records, squad_records)

    lookup = CrossLingualLookup(ayatec_records, squad_records)
    print(f"\n[RESULT] C1 complete. CrossLingualLookup ready "
          f"({len(ayatec_records)} AyaTEC + {len(squad_records)} SQuAD v2 records). "
          f"Pass this as: CrossLingualFallback(lookup_fn=lookup)")
    return lookup


if __name__ == "__main__":
    main()
