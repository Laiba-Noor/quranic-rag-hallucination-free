"""
B3 - Prove the B2 fine-tuning harness works end-to-end using public MTEB Arabic
sample pairs as placeholder data, BEFORE Member A's real verse-Tafsir pairs
arrive at Sync Point 1.

Two modes:
    --dry-run   Uses a tiny, randomly-initialized local transformer (no
                downloads at all) purely to prove the training loop, loss
                wiring, and save/reload cycle work end-to-end. Use this for a
                fast smoke test or when Hugging Face is unreachable.
    (default)   Downloads the real GATE-AraBERT-v1 base model and a real
                Arabic STS sample (MTEB "sts17-crosslingual-sts", ar-ar
                config) to run a short, realistic fine-tuning pass.

Once Member A's real data lands in Sync Point 1, swap `load_placeholder_data()`
in this file for Member A's verse-Tafsir pairs and re-run the exact same
pipeline unchanged - that is the point of building/testing the harness first.

Requirements:
    pip install sentence-transformers torch datasets
"""

import argparse
import random
import sys

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from b2_finetune_harness import HarnessConfig, run_finetuning

random.seed(42)

# Sentences that are clearly NOT Quranic text - used only as unrelated
# negatives / distractors while testing the pipeline mechanics.
OFFLINE_FALLBACK_SENTENCES = [
    ("الطقس جميل اليوم في المدينة", "الجو مشمس ولطيف اليوم"),          # weather is nice / sunny today
    ("أحب قراءة الكتب في المساء", "القراءة في الليل تريحني"),          # I like reading in the evening / reading at night relaxes me
    ("السيارة الجديدة سريعة جدا", "هذه السيارة تسير بسرعة كبيرة"),      # the new car is very fast / this car goes very fast
    ("الطعام في هذا المطعم لذيذ", "وجبات هذا المطعم رائعة الطعم"),      # food at this restaurant is delicious / meals here taste great
    ("الأطفال يلعبون في الحديقة", "الأولاد يمرحون في الحديقة"),          # children are playing in the garden / kids are having fun in the garden
]


def try_load_real_mteb_sample(sample_size: int = 40):
    """
    Attempt to download a small real Arabic STS sample from MTEB
    (sts17-crosslingual-sts, ar-ar config) to use as placeholder pairs.
    Returns None if the dataset cannot be reached (e.g. no network access),
    so callers can fall back to the offline sentence set.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("mteb/sts17-crosslingual-sts", "ar-ar", split="test")
        ds = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))

        # Keep only reasonably similar pairs (score is 0-5 in STS) as positives.
        pairs = [
            {"anchor": row["sentence1"], "positive": row["sentence2"]}
            for row in ds
            if row.get("score", 0) is not None and row["score"] >= 3.0
        ]
        print(f"[OK] Loaded {len(pairs)} real MTEB Arabic (STS17 ar-ar) pairs.")
        return pairs
    except Exception as exc:  # noqa: BLE001
        print(f"[INFO] Could not load real MTEB Arabic sample ({exc}). "
              f"Falling back to offline placeholder sentences.")
        return None


def load_placeholder_data(sample_size: int = 40):
    """
    Return (pairs, triplets) placeholder data for harness testing.
    Tries real MTEB Arabic data first, falls back to a small offline set.
    """
    pairs = try_load_real_mteb_sample(sample_size)
    if not pairs:
        pairs = [{"anchor": a, "positive": b} for a, b in OFFLINE_FALLBACK_SENTENCES]

    all_positives = [p["positive"] for p in pairs]
    triplets = []
    for p in pairs:
        # Sample a negative from another pair's positive sentence.
        negative_candidates = [s for s in all_positives if s != p["positive"]]
        negative = random.choice(negative_candidates)
        triplets.append({"anchor": p["anchor"], "positive": p["positive"], "negative": negative})

    print(f"[OK] Prepared {len(pairs)} pairs and {len(triplets)} triplets for harness testing.")
    return pairs, triplets


def build_dry_run_model(corpus_sentences: list) -> str:
    """
    Build a tiny, randomly-initialized local sentence-transformers model with
    NO downloads at all (no Hugging Face Hub access required): trains a small
    BPE tokenizer on the given corpus and builds a small random-weight BERT
    config from scratch. Used purely to smoke-test the training loop
    mechanics when Hugging Face is unreachable, or as a fast local check.
    Returns a local path that HarnessConfig.base_model_name can point at.
    """
    from tokenizers import Tokenizer, models as tok_models, trainers, pre_tokenizers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
    from sentence_transformers import models

    local_dir = "./_tiny_dry_run_base"

    # 1. Train a minimal BPE tokenizer directly on the local corpus - no download.
    tokenizer = Tokenizer(tok_models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=800,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )
    tokenizer.train_from_iterator(corpus_sentences, trainer=trainer)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    # 2. Build a tiny BERT config from scratch - random weights, no download.
    config = BertConfig(
        vocab_size=fast_tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    tiny_model = BertModel(config)

    tiny_model.save_pretrained(local_dir)
    fast_tokenizer.save_pretrained(local_dir)

    word_embedding_model = models.Transformer(local_dir, max_seq_length=32)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    st_model.save(local_dir)

    print(f"[OK] Built tiny dry-run base model at: {local_dir} "
          f"(vocab={fast_tokenizer.vocab_size}, hidden=32, layers=2) - no network used.")
    return local_dir


def evaluate_similarity_shift(model_path_before: str, model_path_after: str, pairs: list):
    """
    Compare average positive-pair similarity before vs. after fine-tuning as a
    sanity signal that training actually moved the embeddings (not proof of
    quality - just proof the pipeline had an effect).
    """
    anchors = [p["anchor"] for p in pairs]
    positives = [p["positive"] for p in pairs]

    before_model = SentenceTransformer(model_path_before)
    before_a = before_model.encode(anchors, convert_to_tensor=True, normalize_embeddings=True)
    before_p = before_model.encode(positives, convert_to_tensor=True, normalize_embeddings=True)
    before_sim = cos_sim(before_a, before_p).diagonal().mean().item()

    after_model = SentenceTransformer(model_path_after)
    after_a = after_model.encode(anchors, convert_to_tensor=True, normalize_embeddings=True)
    after_p = after_model.encode(positives, convert_to_tensor=True, normalize_embeddings=True)
    after_sim = cos_sim(after_a, after_p).diagonal().mean().item()

    print("\nSimilarity shift check (positive pairs, average cosine similarity):")
    print(f"  before fine-tuning : {before_sim:.4f}")
    print(f"  after  fine-tuning : {after_sim:.4f}")
    print(f"  delta              : {after_sim - before_sim:+.4f}")

    # Save results as a JSON log and a PNG chart in outputs/
    from results_utils import save_json_log, save_bar_chart
    save_json_log("b3_similarity_shift", {
        "before_finetuning": float(before_sim),
        "after_finetuning": float(after_sim),
        "delta": float(after_sim - before_sim),
    })
    save_bar_chart(
        "b3_similarity_shift",
        labels=["before fine-tuning", "after fine-tuning"],
        values=[float(before_sim), float(after_sim)],
        title="B3 - Similarity Shift From Fine-Tuning",
    )

    return before_sim, after_sim


def main():
    parser = argparse.ArgumentParser(description="B3 - Test the B2 harness end-to-end.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Use a tiny local random model, no downloads at all.")
    parser.add_argument("--sample-size", type=int, default=40,
                         help="Number of placeholder pairs to use.")
    args = parser.parse_args()

    pairs, triplets = load_placeholder_data(args.sample_size)

    if args.dry_run:
        corpus_sentences = [p["anchor"] for p in pairs] + [p["positive"] for p in pairs]
        base_model_path = build_dry_run_model(corpus_sentences)
        config = HarnessConfig(
            base_model_name=base_model_path,
            matryoshka_dims=[32, 16, 8],  # small dims to match the tiny model's 32-dim output
            output_dir="./_tiny_dry_run_finetuned",
            num_train_epochs=1,
            per_device_train_batch_size=2,
        )
    else:
        base_model_path = "Omartificial-Intelligence-Space/GATE-AraBert-v1"
        config = HarnessConfig(
            base_model_name=base_model_path,
            output_dir="./gate-arabert-b3-smoketest",
            num_train_epochs=1,
            per_device_train_batch_size=16,
        )

    print(f"\n[STEP] Running fine-tuning harness (dry_run={args.dry_run})...")
    run_finetuning(pairs, triplets, config)

    print("\n[STEP] Evaluating similarity shift before vs. after...")
    ok = evaluate_similarity_shift(base_model_path, config.output_dir, pairs[:10] or pairs)

    print("\n[RESULT] B3 harness test completed successfully.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
