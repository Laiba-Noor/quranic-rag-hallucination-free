# ===== Paste into a SECOND Colab cell (after the verify cell) =====
# Walks every shared folder and reports what's actually in them,
# then diffs their src/ against what GitHub has.

import os, json

ROOT = "/content/drive/MyDrive"
TARGETS = ["Phase1_Project", "Phase2_Project", "Phase3_Project", "qwen_cache"]

# What GitHub's src/ contains, as of commit c5eee27.
GITHUB_SRC = {
    "a2_tafsir_acquisition_arabic.py",
    "a3_preprocess_pipeline_fixed.py",
    "b1_load_model.py",
    "b2_finetune_harness.py",
    "b3_test_harness.py",
    "b4_build_real_triplets.py",
    "b5_finetune_and_benchmark.py",
    "b6_build_index_and_retrieval_api.py",
    "build_final_dataset.py",
    "c1_cross_lingual_fallback.py",
    "c2_sufficiency_labels.py",
    "c3_query_refinement.py",
    "d1_agent_loop.py",
    "d2_cot_prompting.py",
    "d3_sufficiency_and_fallback.py",
    "download_datasets.py",
    "e1_triple_extraction.py",
    "e2_knowledge_graph.py",
    "e3_structural_alignment.py",
    "e4_zero_hallucination_guardrail.py",
    "e5_phase2_integration.py",
    "e6_retrieval_quality_diagnostic.py",
    "generate_phase1_graphs.py",
    "generate_phase2_dashboard.py",
    "generate_phase2_graphs.py",
    "results_utils.py",
    "run_sync_point2.py",
}

print("#" * 72)
print("# PART 1 - FULL TREE (2 levels deep)")
print("#" * 72)
for t in TARGETS:
    base = os.path.join(ROOT, t)
    if not os.path.isdir(base):
        print(f"\n[MISSING] {t}")
        continue
    print(f"\n=== {t} ===")
    for dirpath, dirnames, filenames in os.walk(base):
        depth = dirpath[len(base):].count(os.sep)
        if depth > 2:
            dirnames[:] = []
            continue
        pad = "  " * depth
        print(f"{pad}{os.path.basename(dirpath) or t}/")
        for fn in sorted(filenames)[:40]:
            try:
                mb = os.path.getsize(os.path.join(dirpath, fn)) / 1e6
            except OSError:
                mb = 0
            print(f"{pad}  {fn}  ({mb:.2f} MB)")
        if len(filenames) > 40:
            print(f"{pad}  ... +{len(filenames)-40} more files")

print()
print("#" * 72)
print("# PART 2 - PYTHON FILES IN DRIVE THAT GITHUB DOES NOT HAVE")
print("#" * 72)
found = {}
for t in TARGETS:
    for dirpath, _, filenames in os.walk(os.path.join(ROOT, t)):
        for fn in filenames:
            if fn.endswith(".py"):
                found.setdefault(fn, []).append(os.path.join(dirpath, fn))

extra = sorted(set(found) - GITHUB_SRC)
if extra:
    for fn in extra:
        print(f"\n  NEW  {fn}")
        for p in found[fn][:3]:
            print(f"         {p}")
else:
    print("\n  (none - Drive has no .py files beyond GitHub's)")

print()
print("#" * 72)
print("# PART 3 - THE TWO SCRIPTS WE KNOW ARE MISSING FROM GITHUB")
print("#" * 72)
for want in ["augment_training_with_ayatec.py", "b4_build_real_triplets.py"]:
    hits = found.get(want, [])
    print(f"\n{want}")
    if not hits:
        print("   NOT FOUND in Drive")
    for p in hits:
        n = sum(1 for _ in open(p, encoding="utf-8", errors="replace"))
        head = open(p, encoding="utf-8", errors="replace").read(160).replace("\n", " ")
        print(f"   {p}")
        print(f"      {n} lines | starts: {head[:110]}")

print()
print("#" * 72)
print("# PART 4 - DID THE v2 RETRAIN ACTUALLY FINISH?")
print("#" * 72)
for cand in [
    f"{ROOT}/Phase3_Project/guardrail_output_v2/b5_real_finetuned_v2",
    f"{ROOT}/Phase3_Project/guardrail_output_v2/index_v2",
    f"{ROOT}/Phase3_Project/guardrail_output_v2/phase3_guardrail_results_v2.json",
]:
    print(f"{'YES ' if os.path.exists(cand) else 'no  '} {cand}")
