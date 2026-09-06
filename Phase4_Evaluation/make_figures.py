#!/usr/bin/env python3
"""
Phase 4 — publication figures.

Every number here is transcribed from the saved run artefacts listed in
Phase4_Results.pdf. Nothing is smoothed, rounded up, or invented.

Run:  python3 make_figures.py
Out:  Figures/*.pdf  (vector, for the paper)
      Figures/*.png  (300 dpi, for slides and drafts)

Set TITLES = False for camera-ready versions where the caption carries the
title instead.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

TITLES = True
OUT = "Figures"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.8,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
    "lines.linewidth": 1.6,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,   # embed as TrueType, not Type 3 — required by most venues
    "ps.fonttype": 42,
})

# Okabe-Ito, colour-blind safe
C = {
    "base":   "#0072B2",
    "v1":     "#D55E00",
    "v2":     "#E69F00",
    "bm25":   "#999999",
    "hyb_b":  "#56B4E9",
    "hyb_v2": "#CC79A7",
    "good":   "#009E73",
    "bad":    "#D55E00",
    "ink":    "#222222",
}

def finish(fig, name, title=None):
    if TITLES and title:
        fig.suptitle(title, fontsize=9.5, fontweight="bold", y=1.005)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  {name}.pdf / .png")

def grid(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#BBBBBB")

def barlabels(ax, bars, fmt="{:.3f}", size=6.6, pad=1.5):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width()/2, h),
                    textcoords="offset points", xytext=(0, pad),
                    ha="center", va="bottom", fontsize=size, color="#333333")

# ----------------------------------------------------------------- data
SYS = ["base GATE-AraBert-v1", "hybrid RRF (base+BM25)", "v2 fine-tuned",
       "hybrid RRF (v2+BM25)", "v1 fine-tuned", "BM25 lexical"]
KEY = ["base", "hyb_b", "v2", "hyb_v2", "v1", "bm25"]
SHORT = {"base": "base GATE", "hyb_b": "RRF base+BM25", "v2": "v2 fine-tuned",
         "hyb_v2": "RRF v2+BM25", "v1": "v1 fine-tuned", "bm25": "BM25"}

AYA = {  # R@10, R@20, Hit@10, Hit@20, MRR, NDCG@10
    "base":   [0.1587, 0.2246, 0.5690, 0.6724, 0.3626, 0.1981],
    "hyb_b":  [0.1434, 0.1849, 0.5345, 0.6207, 0.2535, 0.1423],
    "v2":     [0.0884, 0.1244, 0.3103, 0.4253, 0.1388, 0.0808],
    "hyb_v2": [0.0764, 0.1074, 0.2816, 0.3966, 0.1450, 0.0741],
    "v1":     [0.0608, 0.0974, 0.2989, 0.3966, 0.1569, 0.0735],
    "bm25":   [0.0236, 0.0312, 0.1437, 0.1724, 0.0656, 0.0295],
}
QRCD = {
    "base":   [0.0795, 0.0977, 0.5030, 0.5799, 0.3015, 0.1457],
    "hyb_b":  [0.0809, 0.1117, 0.4615, 0.5740, 0.2951, 0.1368],
    "v2":     [0.0588, 0.0732, 0.3077, 0.3905, 0.1939, 0.0924],
    "hyb_v2": [0.0624, 0.0921, 0.3609, 0.4734, 0.2146, 0.0995],
    "v1":     [0.0321, 0.0472, 0.2426, 0.2781, 0.1518, 0.0631],
    "bm25":   [0.0459, 0.0550, 0.3018, 0.3964, 0.1958, 0.0848],
}
M = ["R@10", "R@20", "Hit@10", "Hit@20", "MRR", "NDCG@10"]
CEIL_AYA, CEIL_QRCD = 0.7982, 0.5937

# HitRate@k, AyaTEC, k = 1, 5, 10, 20
HITK = {
    "base":   [0.2529, 0.4770, 0.5690, 0.6724],
    "hyb_b":  [0.1379, 0.3793, 0.5345, 0.6207],
    "v2":     [0.0632, 0.1954, 0.3103, 0.4310],
    "v1":     [0.0977, 0.2069, 0.2989, 0.3966],
    "bm25":   [0.0402, 0.0805, 0.1437, 0.1724],
}
RECK = {
    "base":   [0.0520, 0.1170, 0.1587, 0.2246],
    "hyb_b":  [0.0225, 0.0950, 0.1434, 0.1849],
    "v2":     [0.0202, 0.0614, 0.0884, 0.1257],
    "v1":     [0.0112, 0.0367, 0.0601, 0.0967],
    "bm25":   [0.0049, 0.0137, 0.0236, 0.0312],
}
KS = [1, 5, 10, 20]

# tafsir ablation (v1 unless noted)
ABL = {
    "verses + tafsir":  [0.0422, 0.0601, 0.2126, 0.2644, 0.1258, 0.0561],
    "verses only":      [0.0610, 0.0950, 0.2989, 0.3908, 0.1612, 0.0747],
    "tafsir only":      [0.0120, 0.0160, 0.0517, 0.1034, 0.0237, 0.0142],
}

EF   = [10, 32, 64, 128, 256, 512]
EF_R = [0.1336, 0.1512, 0.1559, 0.1576, 0.1588, 0.1588]
EF_H = [0.4828, 0.5345, 0.5402, 0.5632, 0.5690, 0.5690]
EF_M = [0.3000, 0.3339, 0.3414, 0.3511, 0.3569, 0.3569]
EF_W = [21.3, 6.3, 3.4, 0.6, 0.0, 0.0]

ABST_ROWS = ["Correctly refused\n(unanswerable)", "Over-refused\n(answerable)",
             "Cited a gold verse", "All citations\ngrounded"]
ABST = {
    "naive":          [0.424, 0.217, 0.149, 0.170],
    "+ instruction":  [0.424, 0.350, 0.436, 0.513],
    "+ gate @0.62":   [0.970, 0.883, 0.714, 0.429],
}

AUC_SIG = ["top-1 similarity", "mean of top-5", "count above 0.55",
           "top-1 − top-2 margin", "std. dev. of top-10"]
AUC_VAL = [0.6344, 0.6075, 0.5822, 0.5765, 0.5738]

OPS = [  # label, over-refusal (x), correct refusal (y)
    ("naive RAG",            0.217, 0.424),
    ("+ instruction",        0.350, 0.424),
    ("+ gate @0.62",         0.883, 0.970),
    ("best threshold 0.411", 0.207, 0.424),
]

# structural guardrail, 23 Aug run
GR = [  # mismatch, n_cited, n_ungrounded, verified
    (0.857, 5, 1, False), (0.750, 2, 0, False), (0.762, 3, 1, False),
    (0.448, 5, 0, True),  (1.000, 0, 0, False), (1.000, 4, 1, False),
    (0.938, 5, 5, False), (0.441, 5, 3, True),  (0.909, 5, 1, False),
    (0.917, 2, 0, False),
]

REPRO = [("base ·\nverses only", 5.4e-7), ("v2 ft ·\n+tafsir", 0.286),
         ("v1 ft ·\n+tafsir", 0.347), ("base ·\n+tafsir", 0.358)]

WILC = [("AyaTEC  Recall@10", -0.0703, 9.2e-8, -0.277),
        ("AyaTEC  HitRate@20", -0.2471, 6.0e-8, -0.450),
        ("QRCD    Recall@10", -0.0207, 1.6e-4, -0.116),
        ("QRCD    HitRate@20", -0.1893, 1.6e-4, -0.303)]

print("writing figures to ./Figures/")

# ============================================================ FIG 1
fig, ax = plt.subplots(figsize=(7.0, 3.1))
idx = [2, 3, 4, 5]                      # Hit@10, Hit@20, MRR, NDCG@10
labels = [M[i] for i in idx]
x = np.arange(len(idx)); w = 0.14
for j, k in enumerate(KEY):
    vals = [AYA[k][i] for i in idx]
    b = ax.bar(x + (j - 2.5)*w, vals, w, color=C[k], label=SYS[j],
               edgecolor="white", linewidth=0.4)
    if k == "base":
        barlabels(ax, b, size=6.3)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("score"); ax.set_ylim(0, 0.75)
ax.legend(ncol=3, loc="upper right", columnspacing=1.0, handlelength=1.2)
grid(ax)
finish(fig, "fig01_main_ayatec",
       "Retrieval on AyaTEC (174 questions) — the untrained model leads on every metric")

# ============================================================ FIG 2
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=False)
fig.subplots_adjust(wspace=0.55)
for ax, data, name, ceil in ((axes[0], AYA, "AyaTEC (174 q)", CEIL_AYA),
                             (axes[1], QRCD, "QRCD (169 q)", CEIL_QRCD)):
    order = sorted(KEY, key=lambda k: data[k][4], reverse=True)
    vals = [data[k][4] for k in order]
    names = [SHORT[k] for k in order]
    b = ax.barh(range(len(order))[::-1], vals,
                color=[C[k] for k in order], edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(order))[::-1]); ax.set_yticklabels(names, fontsize=7.6)
    ax.set_xlabel("MRR"); ax.set_title(name, fontsize=9, fontweight="bold")
    ax.set_xlim(0, max(vals)*1.30)
    for r, v in zip(b, vals):
        ax.annotate(f"{v:.3f}", (v, r.get_y()+r.get_height()/2),
                    xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=6.8, color="#333333")
    grid(ax, axis="x")
finish(fig, "fig02_replication_both_benchmarks",
       "The ranking replicates on an independent benchmark")

# ============================================================ FIG 3
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
for ax, D, lab in ((axes[0], HITK, "HitRate@k"), (axes[1], RECK, "Recall@k")):
    for k in ["base", "hyb_b", "v1", "v2", "bm25"]:
        ax.plot(KS, D[k], marker="o", ms=3.6, color=C[k],
                label=SYS[KEY.index(k)], zorder=3)
    ax.set_xscale("log"); ax.set_xticks(KS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("k (log scale)"); ax.set_ylabel(lab)
    grid(ax)
axes[0].legend(loc="upper left", handlelength=1.4)
finish(fig, "fig03_depth_curves",
       "Retrieval quality against cut-off depth, AyaTEC — the gap holds at every k")

# ============================================================ FIG 4
fig, ax = plt.subplots(figsize=(4.6, 2.9))
gain = [(AYA["base"][i] - AYA["v2"][i]) / AYA["v2"][i] * 100 for i in range(6)]
b = ax.bar(M, gain, color=C["good"], edgecolor="white", linewidth=0.4, width=0.62)
for r, v in zip(b, gain):
    ax.annotate(f"+{v:.0f}%", (r.get_x()+r.get_width()/2, v), xytext=(0, 2),
                textcoords="offset points", ha="center", fontsize=7.2,
                fontweight="bold", color="#00674C")
ax.axhline(0, color=C["ink"], lw=0.8)
ax.set_ylabel("improvement of base over v2 fine-tuned (%)")
ax.set_ylim(0, max(gain)*1.2)
grid(ax)
finish(fig, "fig04_relative_improvement",
       "Removing fine-tuning improves every metric (AyaTEC)")

# ============================================================ FIG 5
fig, ax = plt.subplots(figsize=(6.4, 2.9))
x = np.arange(6); w = 0.26
cols = {"verses + tafsir": C["v1"], "verses only": C["good"], "tafsir only": C["bm25"]}
for j, (name, vals) in enumerate(ABL.items()):
    ax.bar(x + (j-1)*w, vals, w, color=cols[name], label=name,
           edgecolor="white", linewidth=0.4)
ax.set_xticks(x); ax.set_xticklabels(M)
ax.set_ylabel("score"); ax.legend(handlelength=1.2)
ax.annotate("", xy=(0-w+0.005, ABL["verses only"][0]), xytext=(0-w+0.005, ABL["verses + tafsir"][0]),
            arrowprops=dict(arrowstyle="->", color="#00674C", lw=1.2))
ax.annotate("+45%", (0.10, (ABL["verses only"][0]+ABL["verses + tafsir"][0])/2),
            fontsize=7.6, fontweight="bold", color="#00674C")
grid(ax)
finish(fig, "fig05_tafsir_ablation",
       "Removing tafsir from the index improves retrieval (v1 model, AyaTEC)")

# ============================================================ FIG 6
fig, ax = plt.subplots(figsize=(4.6, 2.9))
names = ["verses only\n6,236 entries", "verses + tafsir\n12,472 entries",
         "chunked tafsir\n250,356 entries"]
vals  = [0.0610, 0.0422, 0.0197]
b = ax.bar(names, vals, color=[C["good"], C["v1"], C["bad"]],
           edgecolor="white", linewidth=0.4, width=0.6)
barlabels(ax, b, fmt="{:.4f}", size=7.4)
ax.set_ylabel("Recall@10"); ax.set_ylim(0, 0.075)
ax.annotate("p = 8.8×10⁻⁷ vs verses only", (2, 0.0197), xytext=(0, 16),
            textcoords="offset points", ha="center", fontsize=7, color=C["bad"])
grid(ax)
finish(fig, "fig06_chunking_negative",
       "Chunking tafsir to fit the 64-token window makes retrieval worse")

# ============================================================ FIG 7
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
ax = axes[0]
ax.plot(EF, EF_H, marker="o", ms=4, color=C["base"], label="HitRate@10")
ax.plot(EF, EF_M, marker="s", ms=4, color=C["good"], label="MRR@10")
ax.plot(EF, EF_R, marker="^", ms=4, color=C["v2"], label="Recall@10")
ax.axvline(10, color=C["bad"], ls=":", lw=1.1)
ax.annotate("current\ndefault", (10, 0.52), xytext=(6, 0), fontsize=6.8,
            textcoords="offset points", color=C["bad"])
ax.axvline(256, color=C["good"], ls="--", lw=1.1)
ax.annotate("recommended", (256, 0.24), xytext=(-58, 0), fontsize=6.8,
            textcoords="offset points", color="#00674C")
ax.set_xscale("log"); ax.set_xticks(EF)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("ef (search-time candidate list)"); ax.set_ylabel("score")
ax.legend(loc="lower right", handlelength=1.4, bbox_to_anchor=(1.0, 0.02)); grid(ax)

ax = axes[1]
b = ax.bar([str(e) for e in EF], EF_W,
           color=[C["bad"]] + [C["bm25"]]*3 + [C["good"]]*2,
           edgecolor="white", linewidth=0.4, width=0.62)
barlabels(ax, b, fmt="{:.1f}%", size=7)
ax.set_xlabel("ef"); ax.set_ylabel("questions with wrong top-1 verse (%)")
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
ax.set_ylim(0, 25); grid(ax)
finish(fig, "fig07_ef_sweep",
       "An unset search parameter costs ~19% of retrieval accuracy, free to recover")

# ============================================================ FIG 8
fig, ax = plt.subplots(figsize=(4.6, 2.9))
names = [n for n, _ in REPRO]; devs = [d for _, d in REPRO]
b = ax.bar(names, devs, color=[C["good"]] + [C["bm25"]]*3,
           edgecolor="white", linewidth=0.4, width=0.58)
ax.set_yscale("log"); ax.set_ylabel("largest deviation from recorded score")
ax.axhline(1e-5, color=C["ink"], ls="--", lw=0.9)
ax.annotate("float32 noise floor", (3.35, 1.3e-5), ha="right", fontsize=6.8, color="#333333")
for r, v in zip(b, devs):
    ax.annotate(f"{v:.1e}" if v < 1e-3 else f"{v:.3f}",
                (r.get_x()+r.get_width()/2, v), xytext=(0, 2),
                textcoords="offset points", ha="center", fontsize=6.8)
ax.set_ylim(1e-8, 3)
grid(ax)
finish(fig, "fig08_reproduction_check",
       "Verifying which configuration the deployed system runs")

# ============================================================ FIG 9
fig, ax = plt.subplots(figsize=(6.6, 3.0))
x = np.arange(len(ABST_ROWS)); w = 0.26
cols = [C["bm25"], C["base"], C["bad"]]
for j, (name, vals) in enumerate(ABST.items()):
    b = ax.bar(x + (j-1)*w, vals, w, color=cols[j], label=name,
               edgecolor="white", linewidth=0.4)
    barlabels(ax, b, fmt="{:.2f}", size=6.4)
ax.set_xticks(x); ax.set_xticklabels(ABST_ROWS, fontsize=7.4)
ax.set_ylabel("rate"); ax.set_ylim(0, 1.12); ax.legend(ncol=3, handlelength=1.2)
grid(ax)
finish(fig, "fig09_abstention",
       "Abstention: the instruction changes refusal by zero; the gate refuses almost everything")

# ============================================================ FIG 10
fig, ax = plt.subplots(figsize=(4.4, 4.0))
ax.plot([0, 1], [0, 1], color="#AAAAAA", ls="--", lw=0.9, zorder=1)
ax.annotate("no discrimination", (0.62, 0.58), rotation=38, fontsize=6.8,
            color="#888888", ha="center")
mk = ["o", "s", "^", "D"]
cl = [C["bm25"], C["base"], C["bad"], C["good"]]
for (lab, xx, yy), m, c in zip(OPS, mk, cl):
    ax.scatter(xx, yy, s=64, marker=m, color=c, zorder=3,
               edgecolor="white", linewidth=0.8, label=lab)
ax.annotate("naive RAG and the best possible\nthreshold are the same point",
            xy=(0.215, 0.424), xytext=(0.36, 0.60), fontsize=7,
            arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9))
ax.set_xlabel("over-refusal on answerable questions")
ax.set_ylabel("correct refusal on unanswerable questions")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(loc="lower right", bbox_to_anchor=(1.02, -0.02)); grid(ax, axis="both")
ax.set_aspect("equal")
finish(fig, "fig10_operating_points",
       "The sufficiency gate cannot beat the model's own judgement")

# ============================================================ FIG 11
fig, ax = plt.subplots(figsize=(5.0, 2.8))
y = np.arange(len(AUC_SIG))[::-1]
b = ax.barh(y, AUC_VAL, color=[C["base"]] + [C["bm25"]]*4,
            edgecolor="white", linewidth=0.4, height=0.62)
ax.set_yticks(y); ax.set_yticklabels(AUC_SIG, fontsize=7.6)
ax.axvline(0.5, color="#888888", ls="--", lw=1.0)
ax.axvline(0.8, color=C["good"], ls="--", lw=1.0)
ax.annotate("chance", xy=(0.5, 1.02), xycoords=("data", "axes fraction"),
            fontsize=6.8, color="#666666", ha="center")
ax.annotate("usable", xy=(0.8, 1.02), xycoords=("data", "axes fraction"),
            fontsize=6.8, color="#00674C", ha="center")
for r, v in zip(b, AUC_VAL):
    ax.annotate(f"{v:.3f}", (v, r.get_y()+r.get_height()/2), xytext=(3, 0),
                textcoords="offset points", va="center", fontsize=7)
ax.set_xlim(0.4, 0.9); ax.set_xlabel("ROC AUC — answerable vs unanswerable")
grid(ax, axis="x")
finish(fig, "fig11_answerability_auc",
       "No retrieval-score signal separates answerable from unanswerable questions")

# ============================================================ FIG 12
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0),
                         gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
clean = [m for m, c, u, v in GR if c > 0 and u == 0]
dirty = [m for m, c, u, v in GR if c > 0 and u > 0]
rng = np.random.default_rng(3)
for vals, xpos, col, lab in ((clean, 0, C["good"], "all citations grounded"),
                             (dirty, 1, C["bad"], "≥1 ungrounded citation")):
    ax.scatter(np.full(len(vals), xpos) + rng.uniform(-.10, .10, len(vals)),
               vals, s=46, color=col, alpha=.85, edgecolor="white",
               linewidth=.7, zorder=3, label=lab)
    ax.hlines(np.mean(vals), xpos-.26, xpos+.26, color=col, lw=2, zorder=4)
ax.axhline(0.5, color=C["ink"], ls="--", lw=1.0)
ax.annotate("tolerance 0.50 — verified below", xy=(-0.46, 0.5),
            xytext=(0, 3), textcoords="offset points", fontsize=6.8,
            ha="left", color="#333333")
ax.set_xticks([0, 1]); ax.set_xticklabels(["grounded\n(n=3)", "ungrounded\n(n=6)"])
ax.set_ylabel("structural mismatch score"); ax.set_ylim(0, 1.08)
ax.set_xlim(-.5, 1.5); grid(ax)

ax = axes[1]
names = ["0.50\nused in run", "0.00\ncalibrated",
         "0.75\nbest possible", "—\nalways guess"]
vals = [66.7, 66.7, 77.8, 66.7]
b = ax.bar(names, vals, color=[C["bad"], C["bad"], C["bm25"], C["ink"]],
           edgecolor="white", linewidth=0.4, width=0.6)
barlabels(ax, b, fmt="{:.1f}%", size=7)
ax.axhline(66.7, color=C["ink"], ls="--", lw=1.0)
ax.set_ylabel("accuracy vs citation groundedness (%)")
ax.set_ylim(0, 100); ax.tick_params(axis="x", labelsize=7.2)
ax.set_xlabel("mismatch tolerance", fontsize=8)
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
grid(ax)
finish(fig, "fig12_structural_guardrail",
       "The structural guardrail does not separate grounded from ungrounded answers")

# ============================================================ FIG 13
fig, ax = plt.subplots(figsize=(5.4, 2.5))
y = np.arange(len(WILC))[::-1]
deltas = [d for _, d, _, _ in WILC]
ax.barh(y, deltas, color=C["bad"], height=0.5, edgecolor="white", linewidth=0.4)
ax.axvline(0, color=C["ink"], lw=0.9)
ax.set_yticks(y)
ax.set_yticklabels([w[0] for w in WILC], fontsize=7.4, family="monospace")
for yy, (_, d, p, dd) in zip(y, WILC):
    ax.annotate(f"Δ={d:+.4f}   p={p:.1e}   d={dd:.3f}",
                (d, yy), xytext=(-6, 0), textcoords="offset points",
                ha="right", va="center", fontsize=6.9, color="#333333")
ax.set_xlabel("change when fine-tuned v2 replaces base (negative = worse)")
ax.set_xlim(min(deltas)*1.85, 0.02)
grid(ax, axis="x")
finish(fig, "fig13_significance",
       "Paired Wilcoxon: fine-tuning is significantly worse on both benchmarks")

# ============================================================ FIG 14
fig, ax = plt.subplots(figsize=(5.2, 2.9))
sysnames = ["base", "hyb_b", "v2", "v1", "bm25"]
ach = [AYA[k][0] / CEIL_AYA * 100 for k in sysnames]
b = ax.bar([SYS[KEY.index(k)] for k in sysnames], ach,
           color=[C[k] for k in sysnames], edgecolor="white",
           linewidth=0.4, width=0.6)
barlabels(ax, b, fmt="{:.1f}%", size=7)
ax.axhline(100, color=C["ink"], ls="--", lw=1.0)
ax.annotate("ceiling: every gold verse is in the index (Recall@10 = 0.798)",
            (4.4, 100), xytext=(0, 4), textcoords="offset points",
            ha="right", fontsize=6.8, color="#333333")
ax.set_ylabel("share of achievable Recall@10 (%)")
ax.set_ylim(0, 118); ax.tick_params(axis="x", labelsize=6.9, rotation=12)
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
grid(ax)
finish(fig, "fig14_headroom",
       "How much of the achievable recall each system reaches — headroom remains")

print("\ndone —", len([f for f in os.listdir(OUT) if f.endswith('.pdf')]), "figures")
