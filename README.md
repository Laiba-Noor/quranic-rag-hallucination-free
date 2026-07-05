# Member B — Embedding & Indexing Track: B1, B2, B3

Code for the three independent, start-immediately tasks from the Phase 1 Work
Division Plan. No dependency on Member A's data is required to run any of this.

## Setup

```bash
pip install -r requirements.txt
```

## Files

### `src/b1_load_model.py`
Loads the GATE-AraBERT-v1 base model (falls back to Arabic-Triplet-Matryoshka-V2
if unavailable), encodes a small set of Classical Arabic test sentences, and
checks that semantically related sentences score higher in cosine similarity
than an unrelated one — a quick sanity check that the base model works before
any fine-tuning starts.

```bash
python src/b1_load_model.py
```

### `src/b2_finetune_harness.py`
The reusable fine-tuning harness: hybrid loss combining `MultipleNegativesRankingLoss`
(contrastive) and `TripletLoss` (triplet), both wrapped in `MatryoshkaLoss` so
the resulting embeddings stay useful when truncated to smaller dimensions
(768 → 512 → 256 → 128 → 64). Both objectives train jointly via
`SentenceTransformerTrainer`'s multi-dataset / multi-loss support.

This file is data-agnostic — it takes `pairs` and `triplets` as plain Python
lists and works identically whether the data is placeholder (B3) or
Member A's real verse-Tafsir hand-off.

Run its built-in smoke test directly:
```bash
python src/b2_finetune_harness.py
```

### `src/b3_test_harness.py`
Proves the B2 harness works end-to-end using placeholder data, before
Member A's real corpus is ready:
- Tries to pull a small real Arabic sample from MTEB (`sts17-crosslingual-sts`,
  `ar-ar` config); falls back to a small offline Arabic sentence set if the
  Hub is unreachable.
- Builds pairs and triplets from that placeholder data.
- Runs the full B2 harness for one short epoch.
- Reports similarity shift before vs. after fine-tuning as a sanity signal.

```bash
# Full run against the real GATE model + real MTEB sample (needs internet)
python src/b3_test_harness.py

# Fast offline smoke test: tiny local random model, no downloads at all
python src/b3_test_harness.py --dry-run
```

The `--dry-run` mode was used to verify this entire pipeline (tokenizer
training → tiny model → hybrid Matryoshka loss → trainer → save/reload →
similarity check) actually executes without errors, fully offline.

## Saved outputs

`b1_load_model.py` and `b3_test_harness.py` both save their results into an
`outputs/` folder (created automatically next to the script) via
`results_utils.py`:
- A timestamped `.json` file with the exact numbers.
- A timestamped `.png` bar chart visualizing the same numbers.

No images are otherwise produced by this pipeline — these are text-embedding
models, not vision models, so `outputs/` only ever contains these two result
types plus the saved model folders (`config.json`, `model.safetensors`,
tokenizer files) written by the fine-tuning steps themselves.

## Handoff to Sync Point 1

When Member A delivers the verse-Tafsir pairs, replace the call to
`load_placeholder_data()` in `b3_test_harness.py` with Member A's real data
(same `{"anchor": ..., "positive": ...}` / `{"anchor": ..., "positive": ...,
"negative": ...}` format) and re-run `run_finetuning()` from
`b2_finetune_harness.py` — no other code changes needed.
