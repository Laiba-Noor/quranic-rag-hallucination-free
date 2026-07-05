"""
B1 - Load base embedding model and confirm inference works.

Task: Load GATE-AraBERT-v1 (or the Arabic-Triplet-Matryoshka-V2 checkpoint) and
verify it produces sensible embeddings and similarity scores on a handful of
Classical/Quranic Arabic test sentences before any fine-tuning begins.

Requirements:
    pip install sentence-transformers torch

Model cards (verify current names/availability before running):
    https://huggingface.co/Omartificial-Intelligence-Space/GATE-AraBert-v1
    https://huggingface.co/Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2
"""

import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Two candidate base checkpoints. GATE-AraBert-v1 is the primary target from the
# methodology; Arabic-Triplet-Matryoshka-V2 is a fallback that already ships with
# native Matryoshka support and can be used if GATE is unavailable or underperforms.
PRIMARY_MODEL = "Omartificial-Intelligence-Space/GATE-AraBert-v1"
FALLBACK_MODEL = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"

# Small, hand-picked Classical Arabic / Quranic-register test set.
# Pair (0, 1) are semantically related (mercy / forgiveness theme).
# Pair (2, 3) are semantically related (patience / trial theme).
# Sentence 4 is unrelated to all of the above and should score low against them.
TEST_SENTENCES = [
    "إِنَّ اللَّهَ غَفُورٌ رَحِيمٌ",              # 0: Indeed, Allah is Forgiving, Merciful
    "وَرَحْمَتِي وَسِعَتْ كُلَّ شَيْءٍ",            # 1: My mercy encompasses all things
    "وَبَشِّرِ الصَّابِرِينَ",                      # 2: And give good tidings to the patient
    "إِنَّ مَعَ الْعُسْرِ يُسْرًا",                  # 3: Indeed, with hardship comes ease
    "الشَّمْسُ سَاطِعَةٌ الْيَوْمَ",                # 4: The sun is bright today (unrelated, MSA)
]


def load_model(model_name: str) -> SentenceTransformer:
    """Load a sentence-transformers model, raising a clear error on failure."""
    try:
        model = SentenceTransformer(model_name)
        print(f"[OK] Loaded model: {model_name}")
        print(f"     Max sequence length: {model.max_seq_length}")
        print(f"     Native embedding dimension: {model.get_sentence_embedding_dimension()}")
        return model
    except Exception as exc:  # noqa: BLE001 - surfaced intentionally at the CLI
        print(f"[FAIL] Could not load '{model_name}': {exc}", file=sys.stderr)
        raise


def run_inference_check(model: SentenceTransformer) -> np.ndarray:
    """Encode the test sentences and return the embedding matrix."""
    embeddings = model.encode(
        TEST_SENTENCES,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    print(f"[OK] Encoded {len(TEST_SENTENCES)} sentences -> shape {embeddings.shape}")
    return embeddings


def sanity_check_similarity(embeddings: np.ndarray) -> bool:
    """
    Confirm the model captures Quranic-theme semantic similarity correctly:
    related pairs should score noticeably higher than the unrelated sentence.
    """
    sims = cos_sim(embeddings, embeddings).numpy()

    mercy_pair_sim = sims[0][1]
    patience_pair_sim = sims[2][3]
    unrelated_sim_avg = float(np.mean([sims[4][0], sims[4][1], sims[4][2], sims[4][3]]))

    print("\nSimilarity check:")
    print(f"  mercy pair (0,1)      : {mercy_pair_sim:.4f}")
    print(f"  patience pair (2,3)   : {patience_pair_sim:.4f}")
    print(f"  unrelated avg (4,*)   : {unrelated_sim_avg:.4f}")

    passed = (mercy_pair_sim > unrelated_sim_avg) and (patience_pair_sim > unrelated_sim_avg)
    print(f"\n[{'PASS' if passed else 'FAIL'}] Related pairs score higher than unrelated sentence.")

    # Save results as a JSON log and a PNG chart in outputs/
    from results_utils import save_json_log, save_bar_chart
    save_json_log("b1_similarity_check", {
        "mercy_pair_sim": float(mercy_pair_sim),
        "patience_pair_sim": float(patience_pair_sim),
        "unrelated_sim_avg": float(unrelated_sim_avg),
        "passed": bool(passed),
    })
    save_bar_chart(
        "b1_similarity_check",
        labels=["mercy pair", "patience pair", "unrelated avg"],
        values=[float(mercy_pair_sim), float(patience_pair_sim), float(unrelated_sim_avg)],
        title="B1 - Base Model Similarity Check",
        threshold=float(unrelated_sim_avg),
    )

    return passed


def main():
    model = None
    for candidate in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            model = load_model(candidate)
            break
        except Exception:
            print(f"[INFO] Falling back from '{candidate}'...")
            continue

    if model is None:
        print("[FAIL] Neither primary nor fallback model could be loaded. "
              "Check network access and model availability on Hugging Face.", file=sys.stderr)
        sys.exit(1)

    embeddings = run_inference_check(model)
    ok = sanity_check_similarity(embeddings)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
