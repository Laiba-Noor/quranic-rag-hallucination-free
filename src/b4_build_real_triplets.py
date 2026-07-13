"""
results_utils.py - Shared helper to save script output as:
    1. A plain text/JSON log file (exact numbers, timestamped).
    2. A PNG chart image summarizing the results visually.

Both are written into an `outputs/` folder (created automatically) so B1, B2,
and B3 all leave a permanent, shareable record instead of only printing to
the console.

Usage (see integration notes at the bottom of this file):
    from results_utils import save_json_log, save_bar_chart

Requirements:
    pip install matplotlib
"""

import json
import os
from datetime import datetime

OUTPUT_DIR = "outputs"


def _ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json_log(name: str, data: dict) -> str:
    """
    Save a dict of results (numbers, pass/fail, labels) as a timestamped
    JSON file in outputs/. Returns the file path written.
    """
    out_dir = _ensure_output_dir()
    path = os.path.join(out_dir, f"{name}_{_timestamp()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] JSON log: {path}")
    return path


def save_bar_chart(
    name: str,
    labels: list,
    values: list,
    title: str,
    ylabel: str = "Cosine similarity",
    threshold: float = None,
    threshold_label: str = "unrelated baseline",
) -> str:
    """
    Save a simple bar chart image (PNG) comparing labeled numeric results.
    Useful for B1's similarity check and B3's before/after comparison.

    Example:
        save_bar_chart(
            "b1_similarity_check",
            labels=["mercy pair", "patience pair", "unrelated avg"],
            values=[0.81, 0.77, 0.42],
            title="B1 - Base Model Similarity Check",
        )
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_output_dir()
    path = os.path.join(out_dir, f"{name}_{_timestamp()}.png")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bar_colors = ["#2F5C9E" if v >= (threshold or -1) else "#B23A3A" for v in values]
    bars = ax.bar(labels, values, color=bar_colors, edgecolor="#1F3A5F", linewidth=1.2)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9)

    if threshold is not None:
        ax.axhline(threshold, color="#888888", linestyle="--", linewidth=1)
        ax.text(len(labels) - 0.5, threshold + 0.01, threshold_label,
                 ha="right", fontsize=8, color="#555555")

    ax.set_title(title, fontsize=12, weight="bold", color="#1F3A5F")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values + ([threshold] if threshold else [0])) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)

    print(f"[SAVED] Chart image: {path}")
    return path


# ---------------------------------------------------------------------------
# INTEGRATION NOTES
# ---------------------------------------------------------------------------
# In b1_load_model.py, inside sanity_check_similarity(), after computing
# mercy_pair_sim / patience_pair_sim / unrelated_sim_avg, add:
#
#     from results_utils import save_json_log, save_bar_chart
#     save_json_log("b1_similarity_check", {
#         "mercy_pair_sim": mercy_pair_sim,
#         "patience_pair_sim": patience_pair_sim,
#         "unrelated_sim_avg": unrelated_sim_avg,
#         "passed": passed,
#     })
#     save_bar_chart(
#         "b1_similarity_check",
#         labels=["mercy pair", "patience pair", "unrelated avg"],
#         values=[mercy_pair_sim, patience_pair_sim, unrelated_sim_avg],
#         title="B1 - Base Model Similarity Check",
#         threshold=unrelated_sim_avg,
#     )
#
# In b3_test_harness.py, inside evaluate_similarity_shift(), after computing
# before_sim / after_sim, add:
#
#     from results_utils import save_json_log, save_bar_chart
#     save_json_log("b3_similarity_shift", {
#         "before_finetuning": before_sim,
#         "after_finetuning": after_sim,
#         "delta": after_sim - before_sim,
#     })
#     save_bar_chart(
#         "b3_similarity_shift",
#         labels=["before fine-tuning", "after fine-tuning"],
#         values=[before_sim, after_sim],
#         title="B3 - Similarity Shift From Fine-Tuning",
#     )
# ---------------------------------------------------------------------------
