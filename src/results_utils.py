# Member B — B4, B5, B6 (Real Data Fine-Tuning Through Retrieval API)

Completes Member B's side of Phase 1: real data in, fine-tuned model out,
callable retrieval API ready for Phase 2.

## Setup

```bash
pip install -r requirements.txt
```

Run from a folder that also has Member A's `quranNLP/shared/data/final_cross_reference_index.csv`
(the Sync Point 1 handoff file) accessible — either copy it next to these
scripts, or run from your repo root where `quranNLP/` already exists.

## Files

### `src/b4_build_real_triplets.py`
Reads Member A's real `final_cross_reference_index.csv` and builds:
- **Verse–tafsir pairs**: only from rows where `has_direct_tafsir=True`, so
  fallback/inherited tafsir text never gets used as if it were a direct match.
- **Verse–related-verse pairs**: from the `related_verse_keys` column.
- **Triplets**: each pair plus a negative sampled from a *different surah*,
  so the model learns to separate unrelated theological content specifically.

Includes a leakage check (negative never equals positive) before saving.

```bash
python src/b4_build_real_triplets.py
# or, for a fast first test:
python src/b4_build_real_triplets.py --max-triplets 200
```
Output: `data/real_training_pairs.json`, `data/real_training_triplets.json`

### `src/b5_finetune_and_benchmark.py`
Fine-tunes on B4's real data using the exact same hybrid Matryoshka harness
from B2 — no new training code, just real data instead of placeholders. Then
benchmarks:
- QRCD retrieval top-1 accuracy (does the model rank the true passage above
  distractors for real Quranic reading-comprehension questions?)

```bash
# Fast offline smoke test (tiny local model, no downloads)
python src/b5_finetune_and_benchmark.py --dry-run

# Real run against GATE-AraBERT-v1 (needs internet)
python src/b5_finetune_and_benchmark.py --epochs 4
```
Output: fine-tuned model folder (`b5_real_finetuned/`), benchmark results in
`outputs/b5_benchmark_*.json`

### `src/b6_build_index_and_retrieval_api.py`
Builds an HNSW vector index over every verse **and** every direct tafsir
passage, each tagged with provenance metadata (`verse_key`, `surah`, `ayah`,
`source_type`). Exposes `RetrievalAPI.retrieve(query, top_k)` — call this
directly from Phase 2's agentic retrieval loop.

```bash
python src/b6_build_index_and_retrieval_api.py --model-path ./b5_real_finetuned
```
Output: `index/verses.hnsw`, `index/entries.pkl`

I tested B4 → B5 → B6 end-to-end with a realistic sample dataset before
handing this over — every step ran cleanly, including retrieval returning
correct provenance metadata.

## Recommended order

```bash
python src/b4_build_real_triplets.py
python src/b5_finetune_and_benchmark.py --dry-run   # fast sanity check first
python src/b5_finetune_and_benchmark.py             # real run
python src/b6_build_index_and_retrieval_api.py --model-path ./b5_real_finetuned
```

## This completes Member B's Phase 1 tasks

After B6, the only remaining Phase 1 step is **Sync Point 2 (joint)**: sitting
down with Member A to validate retrieval end-to-end on a shared set of test
theological queries, before Phase 2 development begins.
