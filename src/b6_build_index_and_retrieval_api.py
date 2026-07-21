"""
B6 - Build the vector index over the fine-tuned embeddings and expose a
retrieval function with provenance metadata (verse_key, surah, ayah,
source_type), ready for Phase 2's agentic retrieval loop to call directly.

Uses hnswlib (lightweight, no GPU needed for indexing/search) rather than
FAISS, to keep the dependency footprint small - swap for FAISS later if the
corpus grows past what HNSW comfortably handles.

Requirements:
    pip install hnswlib sentence-transformers pandas
"""

import json
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import hnswlib
from sentence_transformers import SentenceTransformer

BASE_DIR = "quranNLP" if os.path.exists("quranNLP") else "."
CROSS_REF_PATH = os.path.join(BASE_DIR, "shared", "data", "final_cross_reference_index.csv")
INDEX_DIR = "index"


def load_corpus_entries() -> list:
    """
    Build the list of retrievable text chunks with provenance metadata:
    one entry per verse, plus one entry per DIRECT (non-fallback) tafsir
    passage, so retrieval can point back to exactly where a result came from.
    """
    if not os.path.exists(CROSS_REF_PATH):
        print(f"[FAIL] {CROSS_REF_PATH} not found. Run Member A's pipeline first.",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CROSS_REF_PATH)
    df["has_direct_tafsir"] = df["has_direct_tafsir"].astype(str) == "True"

    entries = []
    for _, row in df.iterrows():
        surah, ayah = str(row["verse_key"]).split(":")
        entries.append({
            "text": row["clean_verse"],
            "verse_key": row["verse_key"],
            "surah": int(surah),
            "ayah": int(ayah),
            "source_type": "verse",
        })
        if row["has_direct_tafsir"] and isinstance(row["tafsir_passage"], str) and row["tafsir_passage"]:
            entries.append({
                "text": row["tafsir_passage"],
                "verse_key": row["verse_key"],
                "surah": int(surah),
                "ayah": int(ayah),
                "source_type": "tafsir",
            })

    print(f"[OK] Built {len(entries)} retrievable entries "
          f"({sum(1 for e in entries if e['source_type']=='verse')} verses, "
          f"{sum(1 for e in entries if e['source_type']=='tafsir')} tafsir passages).")
    return entries


def build_index(model: SentenceTransformer, entries: list, dim: int = None):
    """Encode all entries and build an HNSW index over the embeddings."""
    texts = [e["text"] for e in entries]
    embeddings = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=True, batch_size=32,
    )

    dim = dim or embeddings.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(entries), ef_construction=200, M=16)
    index.add_items(embeddings, ids=np.arange(len(entries)))
    index.set_ef(50)

    print(f"[OK] Built HNSW index: {len(entries)} items, dim={dim}")
    return index


def save_index(index, entries: list, out_dir: str = INDEX_DIR):
    os.makedirs(out_dir, exist_ok=True)
    index.save_index(os.path.join(out_dir, "verses.hnsw"))
    with open(os.path.join(out_dir, "entries.pkl"), "wb") as f:
        pickle.dump(entries, f)
    print(f"[SAVED] Index and metadata written to {out_dir}/")


def load_index(dim: int, out_dir: str = INDEX_DIR):
    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(os.path.join(out_dir, "verses.hnsw"))
    with open(os.path.join(out_dir, "entries.pkl"), "rb") as f:
        entries = pickle.load(f)
    return index, entries


class RetrievalAPI:
    """
    Callable retrieval function ready for Phase 2's agentic loop:
        results = retrieval_api.retrieve("query text", top_k=5)
    Each result includes the matched text plus full provenance metadata.
    """

    def __init__(self, model: SentenceTransformer, index, entries: list):
        self.model = model
        self.index = index
        self.entries = entries

    def retrieve(self, query: str, top_k: int = 5) -> list:
        query_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        labels, distances = self.index.knn_query(query_emb, k=min(top_k, len(self.entries)))

        results = []
        for idx, dist in zip(labels[0], distances[0]):
            entry = self.entries[idx]
            results.append({
                **entry,
                "similarity": float(1 - dist),  # hnswlib cosine space returns distance = 1 - cos_sim
            })
        return results


def verify_retrieval(api: RetrievalAPI) -> bool:
    """Spot-check retrieval on a couple of well-known theological queries."""
    test_queries = [
        "الرحمة والمغفرة",   # mercy and forgiveness
        "الصبر على البلاء",   # patience through hardship
    ]

    print("\nRetrieval verification:")
    all_ok = True
    for q in test_queries:
        results = api.retrieve(q, top_k=3)
        print(f"\n  Query: {q}")
        for r in results:
            print(f"    [{r['source_type']}] {r['verse_key']} "
                  f"(sim={r['similarity']:.3f}): {r['text'][:60]}...")
        all_ok = all_ok and len(results) > 0

    print(f"\n[{'PASS' if all_ok else 'FAIL'}] Retrieval returned results with provenance metadata.")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="B6 - Build vector index and retrieval API.")
    parser.add_argument("--model-path", default="./b5_real_finetuned",
                         help="Path to the fine-tuned model from B5 "
                              "(or the base GATE model if B5 hasn't run yet).")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"[INFO] {args.model_path} not found, falling back to base model "
              f"Omartificial-Intelligence-Space/GATE-AraBert-v1")
        args.model_path = "Omartificial-Intelligence-Space/GATE-AraBert-v1"

    model = SentenceTransformer(args.model_path)
    entries = load_corpus_entries()

    index = build_index(model, entries)
    save_index(index, entries)

    api = RetrievalAPI(model, index, entries)
    ok = verify_retrieval(api)

    print(f"\n[RESULT] B6 vector index + retrieval API {'PASSED' if ok else 'FAILED'} verification.")
    print("Ready for Phase 2's agentic retrieval loop to call RetrievalAPI.retrieve(query, top_k).")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
