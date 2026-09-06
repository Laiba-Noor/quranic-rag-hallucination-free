# ===== Paste into a fresh Google Colab cell =====
from google.colab import drive
drive.mount('/content/drive')

import os, json

P1 = "/content/drive/MyDrive/Phase1_Project/MemberB_B4_B6_output"

MUST_EXIST = [
    f"{P1}/b5_real_finetuned/model.safetensors",
    f"{P1}/b5_real_finetuned/config.json",
    f"{P1}/data/real_training_pairs.json",
    f"{P1}/data/real_training_triplets.json",
    f"{P1}/index/verses.hnsw",
    f"{P1}/index/entries.pkl",
    "/content/drive/MyDrive/Phase2_Project/Roma_output",
    "/content/drive/MyDrive/Phase2_Project/Laiba_integration_output",
    "/content/drive/MyDrive/Phase3_Project/guardrail_output_v2",
]

print("=" * 60)
ok = True
for p in MUST_EXIST:
    hit = os.path.exists(p)
    ok &= hit
    size = ""
    if hit and os.path.isfile(p):
        size = f"  ({os.path.getsize(p)/1e6:.1f} MB)"
    print(f"{'OK  ' if hit else 'MISS'}  {p}{size}")

# The decisive check: a real run leaves high-numbered checkpoints.
print("=" * 60)
ck = sorted(d for d in os.listdir(f"{P1}/b5_real_finetuned")
            if d.startswith("checkpoint-")) if os.path.exists(f"{P1}/b5_real_finetuned") else []
print("checkpoints:", ck)
steps = [int(c.split("-")[1]) for c in ck] or [0]
print("REAL TRAINING RUN" if max(steps) > 100 else "SMOKE TEST -- WRONG FOLDER")

# How many examples it actually trained on.
tp = f"{P1}/data/real_training_triplets.json"
if os.path.exists(tp):
    with open(tp) as f:
        print("triplets:", len(json.load(f)))

print("=" * 60)
print("ALL GOOD" if ok else "SOMETHING MISSING -- check shortcut names")
