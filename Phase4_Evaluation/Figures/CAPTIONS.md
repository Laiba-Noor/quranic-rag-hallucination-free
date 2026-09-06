# Figure captions — Phase 4

Draft captions, ready to paste. Each states what the figure shows and what the
reader should take from it. Adjust numbering to your paper's order.

Formats: `.pdf` for the paper (vector, TrueType-embedded, no Type 3 — accepted
by ACL/IEEE/ACM). `.png` at 300 dpi for slides and drafts.

To drop the in-figure titles for camera-ready, set `TITLES = False` at the top
of `make_figures.py` and re-run.

---

## Core results

**fig01_main_ayatec** — *Retrieval performance on AyaTEC (174 questions) across
six systems. The untrained base model leads on every metric, ahead of both
fine-tuned variants and of hybrid lexical–dense fusion.*

**fig02_replication_both_benchmarks** — *Mean reciprocal rank by system on both
benchmarks. The ordering is preserved on QRCD, which was not used in any tuning
decision, indicating the result is not an artefact of AyaTEC.*

**fig03_depth_curves** — *HitRate@k and Recall@k against cut-off depth on
AyaTEC. The margin between the base model and the fine-tuned variants is present
at k = 1 and persists to k = 20; it is not an artefact of a particular cut-off.*

**fig13_significance** — *Paired Wilcoxon signed-rank tests comparing the v2
fine-tuned model against the base model. All four comparisons are negative and
significant at p < 10⁻³, with small-to-medium effect sizes.*

**fig04_relative_improvement** — *Relative improvement of the base model over
the v2 fine-tuned model on AyaTEC. Gains range from 58% (HitRate@20) to 161%
(MRR).*

---

## Ablations

**fig05_tafsir_ablation** — *Effect of index composition on retrieval, holding
the model fixed (v1). Removing tafsir passages raises Recall@10 by 45%;
tafsir-only retrieval is close to noise.*

**fig06_chunking_negative** — *Splitting tafsir into 250-character windows that
fit the encoder's 64-token limit produces 250,356 index entries and degrades
Recall@10 by a further factor of three. Truncation is not the cause of the
tafsir penalty.*

**fig14_headroom** — *Share of achievable Recall@10 reached by each system,
where the ceiling is the recall obtainable if every gold verse were ranked in
the top 10. The best system reaches 19.9%: the ranking is clear, absolute
performance is not.*

---

## Index configuration

**fig07_ef_sweep** — *Left: retrieval quality against the HNSW search-time
parameter `ef`. Right: proportion of questions for which approximate search
returns a different nearest verse than exact search. The library default of
`ef = 10`, used throughout the original system, costs roughly 19% of retrieval
accuracy; metrics saturate at `ef = 256`.*

**fig08_reproduction_check** — *Largest deviation between recorded and
re-computed similarity scores for each candidate configuration, on ten queries
(log scale). Only the base model over a verses-only index reproduces the
deployed system's behaviour, at the float32 noise floor.*

---

## Abstention and guardrails

**fig09_abstention** — *Abstention behaviour over 33 unanswerable and 60
answerable questions. The explicit abstention instruction leaves correct
refusal unchanged at 0.424 while raising over-refusal from 0.217 to 0.350; the
similarity gate refuses nearly everything in both classes.*

**fig11_answerability_auc** — *ROC AUC of five retrieval-derived signals as
answerability classifiers over 207 questions. The strongest, top-1 similarity,
reaches 0.634 — far below the ~0.8 required for a deployable gate.*

**fig10_operating_points** — *Correct refusal against over-refusal for each
configuration. The best operating point reachable by any similarity threshold
(0.424 / 0.207) coincides with plain RAG's default behaviour (0.424 / 0.217):
the gate cannot improve on the model's implicit judgement.*

**fig12_structural_guardrail** — *Left: structural mismatch score for answers
whose citations are all grounded versus those with at least one ungrounded
citation; horizontal bars mark class means. Right: accuracy of the guardrail
against citation groundedness under three tolerances, with the majority-class
baseline. No tolerance, including one calibrated on QRCD, exceeds the baseline.*

---

## Suggested placement

| Section | Figures |
|---|---|
| Main retrieval result | 01, 02, 13 |
| Analysis of the result | 03, 04, 14 |
| Ablations | 05, 06 |
| System configuration | 07, 08 |
| Hallucination / abstention | 09, 11, 10, 12 |

If space is tight, the minimum set that carries the argument is
**01, 13, 05, 07, 10, 12** — one figure per claim.

---

## A note on framing

Figures 09–12 are negative results. Present them as measurements, not as
failures: the contribution is that these components were evaluated for the
first time and found not to work, which is information the field does not
currently have. Reviewers respond badly to negative results that are hidden or
softened, and well to negative results that are clearly measured and honestly
bounded — which is why `fig14` (headroom) and the limitations section matter as
much as the wins.
