"""
generate_phase1_graphs.py - Produce summary charts from everything Member B
has actually run so far: B4's dataset stats, B5's training loss curve and
benchmark results, and B6's live retrieval quality on sample queries.

Every chart here is built from REAL saved artifacts on disk (trainer
checkpoints, JSON logs, the actual fine-tuned model) - nothing here is a
placeholder or synthetic figure.

Requirements:
    pip install matplotlib pandas sentence-transformers
"""

import os
import sys
import json
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRAPH_DIR = "graphs"
NAVY = "#1F3A5F"
STEEL = "#4A6FA5"
ORANGE = "#B25E2A"
GREEN = "#2E7D4F"
RED = "#B23A3A"


def _ensure_dir():
    os.makedirs(GRAPH_DIR, exist_ok=True)


def find_latest_checkpoint(model_dir: str) -> str:
    checkpoints = glob.glob(os.path.join(model_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.split("-")[-1]))


def plot_training_loss(model_dir: str = "./b5_real_finetuned"):
    """Chart 1: training loss over steps, from the real trainer_state.json."""
    checkpoint = find_latest_checkpoint(model_dir)
    if not checkpoint:
        print(f"[SKIP] No checkpoint found under {model_dir} - run B5 first.")
        return None

    state_path = os.path.join(checkpoint, "trainer_state.json")
    if not os.path.exists(state_path):
        print(f"[SKIP] {state_path} not found.")
        return None

    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)

    history = [h for h in state.get("log_history", []) if "loss" in h]
    if not history:
        print("[SKIP] No loss entries in trainer_state.json log_history.")
        return None

    steps = [h.get("step", i) for i, h in enumerate(history)]
    losses = [h["loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, losses, color=STEEL, linewidth=1.8, marker="o", markersize=3)
    ax.set_title("B5 - Training Loss Over Steps (real GATE-AraBERT fine-tuning)",
                  fontsize=12, weight="bold", color=NAVY)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (hybrid contrastive + triplet)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(GRAPH_DIR, "01_training_loss.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path} ({len(history)} logged steps)")
    return path


def plot_dataset_sizes(data_dir: str = "data"):
    """Chart 2: real vs. fallback tafsir coverage, pairs/triplets built by B4."""
    pairs_path = os.path.join(data_dir, "real_training_pairs.json")
    triplets_path = os.path.join(data_dir, "real_training_triplets.json")

    if not os.path.exists(pairs_path) or not os.path.exists(triplets_path):
        print(f"[SKIP] {pairs_path} or {triplets_path} not found - run B4 first.")
        return None

    with open(pairs_path, encoding="utf-8") as f:
        pairs = json.load(f)
    with open(triplets_path, encoding="utf-8") as f:
        triplets = json.load(f)

    labels = ["Pairs\n(contrastive)", "Triplets\n(anchor/pos/neg)"]
    values = [len(pairs), len(triplets)]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, values, color=[STEEL, ORANGE], edgecolor=NAVY, linewidth=1.2, width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.01, f"{v:,}",
                 ha="center", va="bottom", fontsize=10, weight="bold")

    ax.set_title("B4 - Real Training Data Built From Member A's Corpus",
                  fontsize=12, weight="bold", color=NAVY)
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.join(GRAPH_DIR, "02_dataset_sizes.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


def plot_benchmark_results(outputs_dir: str = "outputs"):
    """Chart 3: QRCD retrieval accuracy from B5's saved benchmark JSON."""
    benchmark_files = sorted(glob.glob(os.path.join(outputs_dir, "b5_benchmark_*.json")))
    if not benchmark_files:
        print(f"[SKIP] No b5_benchmark_*.json found in {outputs_dir}/ - run B5 first.")
        return None

    with open(benchmark_files[-1], encoding="utf-8") as f:  # most recent
        bench = json.load(f)

    accuracy = bench.get("qrcd_top1_accuracy")
    if accuracy is None:
        print("[SKIP] qrcd_top1_accuracy not present (QRCD data wasn't available during B5).")
        return None

    fig, ax = plt.subplots(figsize=(5, 4.5))
    color = GREEN if accuracy >= 0.5 else (ORANGE if accuracy >= 0.25 else RED)
    ax.bar(["QRCD top-1\nretrieval accuracy"], [accuracy], color=color, edgecolor=NAVY, width=0.4)
    ax.text(0, accuracy + 0.02, f"{accuracy:.1%}", ha="center", fontsize=13, weight="bold")
    ax.axhline(1 / 10, color="#888888", linestyle="--", linewidth=1)  # 10-candidate random baseline
    ax.text(0.3, 1 / 10 + 0.01, "random baseline (1/10 candidates)", fontsize=8, color="#555555")

    ax.set_ylim(0, 1.05)
    ax.set_title("B5 - Fine-Tuned Model Benchmark on Real QRCD Questions",
                  fontsize=12, weight="bold", color=NAVY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.join(GRAPH_DIR, "03_qrcd_benchmark.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


def plot_retrieval_similarity(model_path: str = "./b5_real_finetuned",
                               index_dir: str = "index"):
    """Chart 4: live retrieval similarity scores for a few real theological queries."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from sentence_transformers import SentenceTransformer
        from b6_build_index_and_retrieval_api import load_index, RetrievalAPI
    except ImportError as exc:
        print(f"[SKIP] Could not import retrieval dependencies: {exc}")
        return None

    if not os.path.exists(os.path.join(index_dir, "verses.hnsw")):
        print(f"[SKIP] {index_dir}/verses.hnsw not found - run B6 first.")
        return None

    model = SentenceTransformer(model_path)
    index, entries = load_index(dim=model.get_sentence_embedding_dimension(), out_dir=index_dir)
    api = RetrievalAPI(model, index, entries)

    queries = ["الرحمة والمغفرة", "الصبر على البلاء", "الايمان بالغيب"]
    fig, axes = plt.subplots(1, len(queries), figsize=(5 * len(queries), 4.5), sharey=True)
    if len(queries) == 1:
        axes = [axes]

    for ax, query in zip(axes, queries):
        results = api.retrieve(query, top_k=3)
        labels = [f"{r['verse_key']}\n({r['source_type']})" for r in results]
        sims = [r["similarity"] for r in results]

        ax.bar(labels, sims, color=STEEL, edgecolor=NAVY)
        ax.set_title(query, fontsize=11, weight="bold")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("B6 - Retrieval Similarity for Sample Theological Queries",
                  fontsize=13, weight="bold", color=NAVY)

    path = os.path.join(GRAPH_DIR, "04_retrieval_quality.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Phase 1 result graphs from real artifacts.")
    parser.add_argument("--model-path", default="./b5_real_finetuned",
                         help="Path to the fine-tuned model (from B5).")
    parser.add_argument("--index-dir", default="index",
                         help="Path to the vector index directory (from B6).")
    args = parser.parse_args()

    _ensure_dir()
    print("[STEP] Generating Phase 1 result graphs from real saved artifacts...\n")

    generated = []
    generated.append(plot_dataset_sizes())
    generated.append(plot_training_loss(model_dir=args.model_path))
    generated.append(plot_benchmark_results())
    generated.append(plot_retrieval_similarity(model_path=args.model_path, index_dir=args.index_dir))

    generated = [p for p in generated if p]
    print(f"\n[RESULT] Generated {len(generated)}/4 graphs in {GRAPH_DIR}/")
    for p in generated:
        print(f"  - {p}")

    if len(generated) < 4:
        print("\n[NOTE] Some graphs were skipped because their source data wasn't found. "
              "Make sure B4, B5, and B6 have all been run before generating graphs.")

    return len(generated) > 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
