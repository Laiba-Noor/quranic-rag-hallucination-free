# Phase 4 — Fully Automated Evaluation

This folder contains all evaluation code, results, and figures for **Phase 4** of the Qur'anic RAG Hallucination-Free system.

## Overview

Phase 4 performs a fully automated evaluation of the retrieval and generation pipeline, benchmarking against **AyaTEC** (174 questions) and **QRCD** datasets. It covers:

- **Retrieval quality**: Recall@k, HitRate@k, MRR, NDCG@10
- **Baseline comparisons**: 6 system variants (base, fine-tuned v1/v2, ±tafsir, BM25, hybrid)
- **Ablation studies**: Tafsir contribution, chunking, `ef` parameter sweep
- **Guardrail evaluation**: Abstention, answerability detection (ROC AUC), structural alignment
- **Statistical significance**: Wilcoxon signed-rank test, Cohen's d, 95% CI

## Key Findings

| Finding | Impact |
|---|---|
| Base (unfine-tuned) model outperforms fine-tuned variants | **+58% to +161% MRR** |
| Removing tafsir from index improves retrieval | **+45%** |
| Fixing `ef` parameter bug (was defaulting to 10) | **+18.9% free improvement** |
| All improvements statistically significant | Wilcoxon p < 0.05 |

## Notebooks (Run in Order on Google Colab)

| # | Notebook | Purpose | GPU? | Time |
|---|---|---|---|---|
| 1 | `1_verify_drive.py` | Verify Google Drive file access | No | <1 min |
| 2 | `2_inventory_drive.py` | Full inventory of Drive contents | No | <1 min |
| 3 | `3_Rebuild_Index_v2.ipynb` | Rebuild lost v2 search index | T4 | 15–40 min |
| 4 | `4_Diagnose_Retrieval.ipynb` | Diagnose retrieval issues | T4 | ~5 min |
| 5 | `5_Phase4_Retrieval_Eval.ipynb` | Full retrieval evaluation | T4 | ~10 min |
| 6 | `6_Fix_And_Chunk.ipynb` | Test chunking strategy | T4 | ~10 min |
| 7 | `7_Fair_Comparison.ipynb` | Fair comparison of all 6 systems | T4 | ~15 min |
| 8 | `8_Baselines.ipynb` | BM25 + hybrid baselines | T4 | ~10 min |
| 9 | `9_Validate_Baseline.ipynb` | Cross-validate baseline results | T4 | ~5 min |
| 10 | `10_Abstention.ipynb` | Abstention & guardrail evaluation | T4 | ~10 min |
| 11 | `11_Answerability.ipynb` | Answerability detection (AUC) | No | ~5 min |
| 12 | `12_Inventory_After_Rerun.ipynb` | Post-rerun inventory check | No | <1 min |
| 13 | `13_What_Did_The_Rerun_Use.ipynb` | Verify rerun model/index config | No | ~2 min |
| 14 | `14_Verify_The_Rerun.ipynb` | Reproduce rerun results | No | 5–8 min |
| 15 | `15_Close_The_Loop.ipynb` | Resolve remaining discrepancies | No | ~5 min |
| 16 | `16_The_EF_Bug.ipynb` | `ef` parameter sweep analysis | No | ~4 min |

### Setup

1. Upload notebook to [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU** (for notebooks 3–10)
3. Add your **Groq API key** as a Colab Secret named `GROQ_API_KEY`
4. Run all cells

## Figures

The `Figures/` subfolder contains **14 publication-ready figures** (PDF vector + PNG 300 dpi):

| Figure | Description |
|---|---|
| `fig01` | Base model leads on all 4 retrieval metrics (AyaTEC) |
| `fig02` | Replication on QRCD benchmark |
| `fig03` | Recall gap holds at every retrieval depth k |
| `fig04` | Relative improvement: +58% to +161% over fine-tuned v2 |
| `fig05` | Tafsir ablation: +45% improvement |
| `fig06` | Chunking ablation (negative result) |
| `fig07` | `ef` parameter sweep: +18.9% free improvement |
| `fig08` | Reproduction verification |
| `fig09` | Abstention analysis |
| `fig10` | Operating point trade-offs |
| `fig11` | Answerability AUC analysis |
| `fig12` | Structural guardrail evaluation |
| `fig13` | Statistical significance (Wilcoxon) |
| `fig14` | Headroom analysis — 19.9% of achievable ceiling |

Draft captions are in `Figures/CAPTIONS.md`.

To regenerate: `python3 make_figures.py` (requires matplotlib).

## Results Report

- `Phase4_Results.html` — Full interactive results report
- `Phase4_Results.pdf` — Print-ready PDF version (6 pages)

## Metrics Glossary

| Metric | Meaning |
|---|---|
| **Recall@k** | Fraction of relevant verses found in top-k results |
| **HitRate@k** | Whether *any* relevant verse appears in top-k |
| **MRR** | Mean Reciprocal Rank — how high the first relevant result ranks |
| **NDCG@10** | Normalized Discounted Cumulative Gain — position-weighted relevance |
