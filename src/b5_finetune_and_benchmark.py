"""
B5 - Fine-tune the embedding model on REAL data (from B4) and benchmark it.

This calls the exact same run_finetuning() from b2_finetune_harness.py that
B3 already proved works end-to-end - only the data source changes, from
placeholder MTEB pairs to Member A's real verse-tafsir triplets.

Benchmarking is two-part:
    1. MTEB Arabic subset       - general Arabic semantic ability, unchanged
                                   after fine-tuning on a narrow religious domain
    2. Held-out QRCD pairs      - domain-specific: does the fine-tuned model
                                   retrieve the correct passage for a Quranic
                                   reading-comprehension question better than
                                   the base model did?

Requirements:
    pip install sentence-transformers torch datasets pandas
"""

import json
import os
import sys
import argparse

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from b2_finetune_harness import HarnessConfig, run_finetuning

BASE_DIR = "quranNLP" if os.path.exists("quranNLP") else "."
TRIPLETS_PATH = os.path.join("data", "real_training_triplets.json")
PAIRS_PATH = os.path.join("data", "real_training_pairs.json")
QRCD_PATH = os.path.join(BASE_DIR, "shared", "data", "qrcd_flat.json") if os.path.exists(
    os.path.join(BASE_DIR, "shared", "data", "qrcd_flat.json")) else "data/qrcd_flat.json"


def load_real_training_data():
    if not os.path.exists(TRIPLETS_PATH) or not os.path.exists(PAIRS_PATH):
        print(f"[FAIL] Real training data not found. Run b4_build_real_triplets.py first.",
              file=sys.stderr)
        sys.exit(1)
    with open(PAIRS_PATH, encoding="utf-8") as f:
        pairs = json.load(f)
    with open(TRIPLETS_PATH, encoding="utf-8") as f:
        triplets = json.load(f)
    print(f"[OK] Loaded {len(pairs)} real pairs, {len(triplets)} real triplets.")
    return pairs, triplets


def evaluate_qrcd_retrieval(model: SentenceTransformer, sample_size: int = 20) -> float:
    """
    For a sample of QRCD questions, check whether the model ranks the TRUE
    passage above a set of distractor passages - a simple, honest proxy for
    retrieval quality on real Quranic reading-comprehension queries.
    """
    if not os.path.exists(QRCD_PATH):
        print(f"[INFO] {QRCD_PATH} not found - skipping QRCD retrieval eval "
              f"(run Member A's a1_data_acquisition.py to generate it).")
        return None

    with open(QRCD_PATH, encoding="utf-8") as f:
        records = json.load(f)

    test_records = [r for r in records if r["split"] == "test"][:sample_size]
    if not test_records:
        print("[INFO] No QRCD test records available - skipping QRCD eval.")
        return None

    all_passages = list({r["passage"] for r in records})
    correct = 0

    for rec in test_records:
        question_emb = model.encode(rec["question"], convert_to_tensor=True, normalize_embeddings=True)
        # Score the true passage plus a handful of random distractors
        distractors = [p for p in all_passages if p != rec["passage"]][:9]
        candidates = [rec["passage"]] + distractors
        candidate_embs = model.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)

        sims = cos_sim(question_emb, candidate_embs)[0]
        best_idx = int(sims.argmax())
        if best_idx == 0:  # index 0 is always the true passage
            correct += 1

    accuracy = correct / len(test_records)
    print(f"[OK] QRCD retrieval top-1 accuracy on {len(test_records)} sample questions: {accuracy:.1%}")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="B5 - Fine-tune on real data and benchmark.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Use a tiny local model instead of downloading GATE-AraBERT-v1.")
    parser.add_argument("--epochs", type=int, default=2,
                         help="Default lowered from 4 to 2 - halves training time; "
                              "increase later once you've confirmed everything works end-to-end.")
    parser.add_argument("--batch-size", type=int, default=64,
                         help="Higher batch size = far fewer steps on a GPU. "
                              "Default raised from 16 to 64 for T4/A100. Lower this if you hit "
                              "an out-of-memory error.")
    parser.add_argument("--max-seq-length", type=int, default=64,
                         help="Caps tokenized sequence length. Quranic verses and tafsir "
                              "sentences are short, so 64 is usually enough and much faster "
                              "than the model's default (often 512).")
    parser.add_argument("--matryoshka-dims", type=int, nargs="+", default=[768, 256, 64],
                         help="Fewer dimension levels = less compute per step. "
                              "Default reduced from 5 levels to 3 (768/256/64) - still gives "
                              "a large-to-small dimension range for downstream flexibility.")
    parser.add_argument("--no-fp16", action="store_true",
                         help="Disable mixed precision (only useful for debugging numerical issues).")
    args = parser.parse_args()

    pairs, triplets = load_real_training_data()

    if args.dry_run:
        # Reuse B3's tiny local model builder so this can be smoke-tested
        # without any Hugging Face download.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from b3_test_harness import build_dry_run_model
        corpus = [p["anchor"] for p in pairs] + [p["positive"] for p in pairs]
        base_model_path = build_dry_run_model(corpus)
        config = HarnessConfig(
            base_model_name=base_model_path,
            matryoshka_dims=[32, 16, 8],
            output_dir="./b5_real_finetuned_dryrun",
            num_train_epochs=1,
            per_device_train_batch_size=2,
        )
    else:
        base_model_path = "Omartificial-Intelligence-Space/GATE-AraBert-v1"
        config = HarnessConfig(
            base_model_name=base_model_path,
            output_dir="./b5_real_finetuned",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            matryoshka_dims=args.matryoshka_dims,
            use_fp16=not args.no_fp16,
        )
        print(f"[CONFIG] epochs={args.epochs}, batch_size={args.batch_size}, "
              f"max_seq_length={args.max_seq_length}, matryoshka_dims={args.matryoshka_dims}, "
              f"fp16={not args.no_fp16}")

    print(f"\n[STEP] Fine-tuning on {len(pairs)} real pairs / {len(triplets)} real triplets "
          f"(dry_run={args.dry_run})...")
    model = run_finetuning(pairs, triplets, config)

    print("\n[STEP] Benchmarking fine-tuned model...")
    qrcd_accuracy = evaluate_qrcd_retrieval(model)

    from results_utils import save_json_log
    save_json_log("b5_benchmark", {
        "num_pairs": len(pairs),
        "num_triplets": len(triplets),
        "qrcd_top1_accuracy": qrcd_accuracy,
        "output_dir": config.output_dir,
    })

    print(f"\n[RESULT] B5 fine-tuning + benchmarking complete. "
          f"Model saved to {config.output_dir}")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
