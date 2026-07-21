"""
generate_phase2_graphs.py - Produce summary charts from the real Phase 2
sync-point integration test (phase2_test_results.json, saved by the
integration notebook's step 10/11).

Every chart here is built from REAL saved results - the actual queries you
ran, actual sufficiency scores, actual fallback triggers, actual grounding
check outcomes.

Requirements:
    pip install matplotlib
"""

import os
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_PATH = "phase2_test_results.json"
GRAPH_DIR = "graphs"
NAVY = "#1F3A5F"
STEEL = "#4A6FA5"
ORANGE = "#B25E2A"
GREEN = "#2E7D4F"
RED = "#B23A3A"


def _ensure_dir():
    os.makedirs(GRAPH_DIR, exist_ok=True)


def load_results(path: str = RESULTS_PATH) -> list:
    if not os.path.exists(path):
        print(f"[FAIL] {path} not found. Run the Phase 2 integration notebook "
              f"through step 11 (which saves this file) first.", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    print(f"[OK] Loaded {len(results)} real test query results.")
    return results


def shorten(text: str, max_len: int = 18) -> str:
    return text if len(text) <= max_len else text[:max_len - 1] + "\u2026"


def plot_rounds_per_query(results: list):
    """Chart 1: how many retrieval rounds each query took (1 = answered immediately, >1 = agent retried)."""
    labels = [shorten(r["query"]) for r in results]
    rounds = [r["rounds"] for r in results]
    colors = [GREEN if r == 1 else ORANGE for r in rounds]

    fig, ax = plt.subplots(figsize=(max(6, len(results) * 1.6), 4.5))
    bars = ax.bar(labels, rounds, color=colors, edgecolor=NAVY)
    for bar, v in zip(bars, rounds):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, str(v), ha="center", fontsize=10, weight="bold")

    ax.set_title("Retrieval Rounds Per Query (D1 Agent Loop)", fontsize=12, weight="bold", color=NAVY)
    ax.set_ylabel("Rounds used")
    ax.set_ylim(0, max(rounds) + 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=20, ha="right")

    path = os.path.join(GRAPH_DIR, "01_rounds_per_query.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


def plot_fallback_triggers(results: list):
    """Chart 2: which queries triggered the cross-lingual fallback."""
    labels = [shorten(r["query"]) for r in results]
    used = [1 if r["used_fallback"] else 0 for r in results]
    colors = [ORANGE if u else STEEL for u in used]

    fig, ax = plt.subplots(figsize=(max(6, len(results) * 1.6), 4.2))
    ax.bar(labels, used, color=colors, edgecolor=NAVY)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"])
    ax.set_title("Cross-Lingual Fallback Triggered? (C1 Integration)", fontsize=12, weight="bold", color=NAVY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=20, ha="right")

    path = os.path.join(GRAPH_DIR, "02_fallback_triggers.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


def plot_grounding_results(results: list):
    """Chart 3: pass/fail rate of the citation grounding check (D2)."""
    passed = sum(1 for r in results if r.get("grounding_ok"))
    failed = sum(1 for r in results if r.get("grounding_ok") is False)
    unknown = len(results) - passed - failed

    labels, values, colors = [], [], []
    if passed: labels.append("Grounded\n(PASS)"); values.append(passed); colors.append(GREEN)
    if failed: labels.append("Hallucinated\ncitation (FAIL)"); values.append(failed); colors.append(RED)
    if unknown: labels.append("No CoT\nresult"); values.append(unknown); colors.append("#AAAAAA")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(labels, values, color=colors, edgecolor=NAVY, width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, str(v), ha="center", fontsize=12, weight="bold")

    ax.set_title("D2 Citation Grounding Check Results", fontsize=12, weight="bold", color=NAVY)
    ax.set_ylabel("Number of queries")
    ax.set_ylim(0, len(results) + 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.join(GRAPH_DIR, "03_grounding_results.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


def plot_summary_table(results: list):
    """Chart 4: a compact visual summary table of every query's outcome."""
    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.5 * len(results)))
    ax.axis("off")

    col_labels = ["Query", "Rounds", "Fallback", "Grounded"]
    cell_text = []
    for r in results:
        cell_text.append([
            shorten(r["query"], 30),
            str(r["rounds"]),
            "Yes" if r["used_fallback"] else "No",
            "PASS" if r.get("grounding_ok") else "FAIL",
        ])

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#F2F2F2" if row % 2 == 0 else "white")
            if col == 3:  # Grounded column
                cell.set_text_props(
                    color=(GREEN if cell_text[row - 1][3] == "PASS" else RED), weight="bold"
                )

    ax.set_title("Phase 2 Sync-Point Integration \u2014 Summary", fontsize=13, weight="bold", color=NAVY, pad=20)

    path = os.path.join(GRAPH_DIR, "04_summary_table.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


def main():
    _ensure_dir()
    results = load_results()

    generated = [
        plot_rounds_per_query(results),
        plot_fallback_triggers(results),
        plot_grounding_results(results),
        plot_summary_table(results),
    ]

    print(f"\n[RESULT] Generated {len(generated)}/4 graphs in {GRAPH_DIR}/")
    for p in generated:
        print(f"  - {p}")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
