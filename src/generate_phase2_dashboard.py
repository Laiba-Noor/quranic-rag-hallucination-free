"""
generate_phase2_dashboard.py - Produce a single, publication-quality dashboard
figure summarizing the Phase 2 sync-point integration test, suitable for a
research paper or presentation. Reads the same real phase2_test_results.json
your notebook already saves - no synthetic data.

Requirements:
    pip install matplotlib numpy
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_PATH = "phase2_test_results.json"
OUTPUT_PATH = "graphs/phase2_dashboard.png"

# Consistent academic color palette (same family used across the project's
# other documents/charts, for visual consistency in a paper or report)
NAVY = "#1F3A5F"
STEEL = "#4A6FA5"
LIGHT_STEEL = "#A9C0DE"
ORANGE = "#B25E2A"
LIGHT_ORANGE = "#E8B98C"
GREEN = "#2E7D4F"
LIGHT_GREEN = "#A8D5BA"
RED = "#B23A3A"
LIGHT_RED = "#E8A5A5"
GREY = "#6B6B6B"
BG = "#FAFAFA"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})


def load_results(path: str = RESULTS_PATH) -> list:
    if not os.path.exists(path):
        print(f"[FAIL] {path} not found.", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    print(f"[OK] Loaded {len(results)} real test query results.")
    return results


def shorten(text: str, max_len: int = 16) -> str:
    return text if len(text) <= max_len else text[:max_len - 1] + "\u2026"


def build_dashboard(results: list, output_path: str = OUTPUT_PATH):
    n = len(results)
    rounds = [r["rounds"] for r in results]
    fallback_flags = [bool(r["used_fallback"]) for r in results]
    grounded_flags = [r.get("grounding_ok") for r in results]
    suff_scores = [r.get("sufficiency_score") for r in results if r.get("sufficiency_score") is not None]
    labels_short = [shorten(r["query"]) for r in results]

    n_grounded = sum(1 for g in grounded_flags if g is True)
    n_ungrounded = sum(1 for g in grounded_flags if g is False)
    n_fallback = sum(fallback_flags)
    avg_rounds = np.mean(rounds) if rounds else 0
    avg_suff = np.mean(suff_scores) if suff_scores else 0

    fig = plt.figure(figsize=(14, 9), facecolor="white")
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.1, 1], hspace=0.55, wspace=0.4)

    fig.suptitle("Phase 2 \u2014 Agentic RAG Pipeline: Sync-Point Integration Results",
                 fontsize=17, weight="bold", color=NAVY, y=0.985)
    fig.text(0.5, 0.955, f"n = {n} real test queries \u00b7 GATE-AraBERT retrieval + Groq Llama-3.3-70B reasoning",
              ha="center", fontsize=10.5, color=GREY, style="italic")

    # Panel 1: Sufficiency score per query
    ax1 = fig.add_subplot(gs[0, 0:2])
    bar_colors = [GREEN if r.get("sufficiency_score", 0) and r["sufficiency_score"] >= 0.5 else ORANGE for r in results]
    scores_plot = [r.get("sufficiency_score") or 0 for r in results]
    bars = ax1.bar(range(n), scores_plot, color=bar_colors, edgecolor=NAVY, linewidth=1.1, width=0.6)
    for i, v in enumerate(scores_plot):
        ax1.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5, weight="bold", color="#222")
    ax1.axhline(avg_suff, color=STEEL, linestyle="--", linewidth=1.3, alpha=0.8)
    ax1.text(n - 0.4, avg_suff + 0.02, f"mean = {avg_suff:.2f}", fontsize=8.5, color=STEEL, ha="right")
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(labels_short, rotation=32, ha="right", fontsize=8.5)
    ax1.set_ylabel("Sufficiency score", fontsize=10)
    ax1.set_title("Retrieval Sufficiency by Query", fontsize=11.5, weight="bold", color=NAVY, loc="left")
    ax1.set_ylim(0, max(scores_plot + [0.1]) * 1.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_facecolor(BG)

    # Panel 2: Grounding check donut
    ax2 = fig.add_subplot(gs[0, 2])
    ground_vals = [v for v in [n_grounded, n_ungrounded] if v > 0]
    ground_labels = [l for l, v in zip(["Grounded", "Ungrounded"], [n_grounded, n_ungrounded]) if v > 0]
    ground_colors = [c for c, v in zip([GREEN, RED], [n_grounded, n_ungrounded]) if v > 0]
    wedges, _, autotexts = ax2.pie(
        ground_vals, colors=ground_colors, startangle=90, counterclock=False,
        autopct=lambda p: f"{int(round(p * sum(ground_vals) / 100))}", pctdistance=0.75,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=11, weight="bold", color="white"),
    )
    ax2.set_title("Citation Grounding\n(D2 guardrail check)", fontsize=11.5, weight="bold", color=NAVY)
    ax2.legend(wedges, ground_labels, loc="lower center", bbox_to_anchor=(0.5, -0.32),
               fontsize=8.5, frameon=False, ncol=1)

    # Panel 3: Rounds per query
    ax3 = fig.add_subplot(gs[1, 0])
    round_colors = [GREEN if rd == 1 else ORANGE for rd in rounds]
    bars3 = ax3.bar(range(n), rounds, color=round_colors, edgecolor=NAVY, linewidth=1.0, width=0.55)
    ax3.set_xticks(range(n))
    ax3.set_xticklabels(labels_short, rotation=32, ha="right", fontsize=8)
    ax3.set_ylabel("Retrieval rounds", fontsize=10)
    ax3.set_title(f"D1 Loop Iterations (mean={avg_rounds:.1f})", fontsize=11, weight="bold", color=NAVY, loc="left")
    ax3.set_ylim(0, max(rounds) + 1)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.set_facecolor(BG)

    # Panel 4: Fallback trigger donut
    ax4 = fig.add_subplot(gs[1, 1])
    fb_vals = [n - n_fallback, n_fallback]
    fb_labels = ["Direct answer", "Cross-lingual\nfallback used"]
    fb_colors = [STEEL, ORANGE]
    fb_vals_nonzero = [v for v in fb_vals if v > 0]
    fb_labels_nonzero = [l for l, v in zip(fb_labels, fb_vals) if v > 0]
    fb_colors_nonzero = [c for c, v in zip(fb_colors, fb_vals) if v > 0]
    wedges4, _, autotexts4 = ax4.pie(
        fb_vals_nonzero, colors=fb_colors_nonzero, startangle=90, counterclock=False,
        autopct=lambda p: f"{int(round(p * sum(fb_vals_nonzero) / 100))}", pctdistance=0.75,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=11, weight="bold", color="white"),
    )
    ax4.set_title("C1 Cross-Lingual\nFallback Usage", fontsize=11.5, weight="bold", color=NAVY)
    ax4.legend(wedges4, fb_labels_nonzero, loc="lower center", bbox_to_anchor=(0.5, -0.38),
               fontsize=8.5, frameon=False, ncol=1)

    # Panel 5: Summary stats box
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    stats_lines = [
        ("Total queries tested", f"{n}"),
        ("Mean retrieval rounds", f"{avg_rounds:.2f}"),
        ("Mean sufficiency score", f"{avg_suff:.2f}"),
        ("Grounding pass rate", f"{n_grounded}/{n_grounded + n_ungrounded} "
                                  f"({(n_grounded / max(n_grounded + n_ungrounded, 1)):.0%})"),
        ("Fallback trigger rate", f"{n_fallback}/{n} ({n_fallback / max(n,1):.0%})"),
    ]
    ax5.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax5.transAxes,
                                  facecolor=NAVY, edgecolor="none", zorder=0))
    ax5.text(0.5, 0.92, "Summary", transform=ax5.transAxes, ha="center",
              fontsize=12.5, weight="bold", color="white")
    y = 0.72
    for label, value in stats_lines:
        ax5.text(0.08, y, label, transform=ax5.transAxes, fontsize=9.3, color="#D8E2F0")
        ax5.text(0.92, y, value, transform=ax5.transAxes, fontsize=10.5, weight="bold",
                  color="white", ha="right")
        y -= 0.17

    fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.12, hspace=0.55, wspace=0.4)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {output_path}")
    return output_path


def main():
    results = load_results()
    build_dashboard(results)
    print("\n[RESULT] Dashboard generated.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
