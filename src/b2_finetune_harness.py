"""
B2 - Hybrid loss (Contrastive + Triplet) fine-tuning harness with
Matryoshka multi-scale optimization.

This module is deliberately data-agnostic: it accepts any correctly formatted
pairs / triplets dataset (real Quranic verse-Tafsir pairs from Member A, or the
placeholder MTEB Arabic data used in B3) and runs the same tested pipeline.

Design:
    - Contrastive objective:  MultipleNegativesRankingLoss on (anchor, positive) pairs.
    - Triplet objective:      TripletLoss on (anchor, positive, negative) triplets.
    - Both objectives are wrapped in MatryoshkaLoss so a single fine-tuning run
      produces embeddings that stay useful when truncated to smaller dimensions
      (768 -> 512 -> 256 -> 128 -> 64), letting downstream retrieval trade off
      accuracy against index size / speed without retraining.
    - Both objectives are trained jointly via SentenceTransformerTrainer's
      multi-dataset / multi-loss support (round-robin sampling across tasks).

Requirements:
    pip install sentence-transformers torch datasets
"""

from dataclasses import dataclass, field
from typing import List, Optional

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers import losses
from sentence_transformers.training_args import BatchSamplers

DEFAULT_MATRYOSHKA_DIMS = [768, 512, 256, 128, 64]


@dataclass
class HarnessConfig:
    base_model_name: str = "Omartificial-Intelligence-Space/GATE-AraBert-v1"
    matryoshka_dims: List[int] = field(default_factory=lambda: DEFAULT_MATRYOSHKA_DIMS.copy())
    output_dir: str = "./gate-arabert-quranic-matryoshka"
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    contrastive_weight: float = 1.0
    triplet_weight: float = 1.0
    triplet_margin: float = 0.5
    seed: int = 42
    max_seq_length: Optional[int] = None  # e.g. 64 - caps tokenization cost per batch
    use_fp16: bool = True  # mixed precision - roughly 2x speedup on GPU, ignored on CPU
    dataloader_num_workers: int = 2


def build_model(config: HarnessConfig) -> SentenceTransformer:
    """Load the base checkpoint that will be fine-tuned."""
    model = SentenceTransformer(config.base_model_name)
    if config.max_seq_length:
        model.max_seq_length = config.max_seq_length
        print(f"[OK] Capped max_seq_length to {config.max_seq_length} "
              f"(shorter sequences = faster training, and Quranic verses/tafsir "
              f"sentences rarely need the model's full default length).")
    print(f"[OK] Base model loaded: {config.base_model_name} "
          f"(native dim = {model.get_sentence_embedding_dimension()})")
    return model


def build_losses(model: SentenceTransformer, config: HarnessConfig) -> dict:
    """
    Build the two Matryoshka-wrapped objectives that together form the
    hybrid contrastive + triplet loss described in the methodology.

    Returns a dict keyed by dataset name, matching the dataset dict produced
    by `build_datasets`, as required by SentenceTransformerTrainer's
    multi-task training mode.
    """
    contrastive_base = losses.MultipleNegativesRankingLoss(model)
    triplet_base = losses.TripletLoss(model, triplet_margin=config.triplet_margin)

    contrastive_matryoshka = losses.MatryoshkaLoss(
        model=model,
        loss=contrastive_base,
        matryoshka_dims=config.matryoshka_dims,
    )
    triplet_matryoshka = losses.MatryoshkaLoss(
        model=model,
        loss=triplet_base,
        matryoshka_dims=config.matryoshka_dims,
    )

    print(f"[OK] Built hybrid loss: contrastive (weight={config.contrastive_weight}) "
          f"+ triplet (weight={config.triplet_weight}), "
          f"Matryoshka dims={config.matryoshka_dims}")

    return {
        "contrastive": contrastive_matryoshka,
        "triplet": triplet_matryoshka,
    }


def build_datasets(
    pairs: List[dict],
    triplets: List[dict],
) -> dict:
    """
    Convert raw Python lists into the Dataset objects the trainer expects.

    pairs:    list of {"anchor": str, "positive": str}
              e.g. {"anchor": verse_text, "positive": matching_tafsir_passage}
    triplets: list of {"anchor": str, "positive": str, "negative": str}
              e.g. {"anchor": verse_text, "positive": related_verse, "negative": unrelated_verse}
    """
    if not pairs:
        raise ValueError("`pairs` is empty - contrastive objective needs at least one pair.")
    if not triplets:
        raise ValueError("`triplets` is empty - triplet objective needs at least one triplet.")

    contrastive_ds = Dataset.from_list(pairs)
    triplet_ds = Dataset.from_list(triplets)

    print(f"[OK] Built datasets: contrastive={len(contrastive_ds)} pairs, "
          f"triplet={len(triplet_ds)} triplets")

    return {
        "contrastive": contrastive_ds,
        "triplet": triplet_ds,
    }


def build_trainer(
    model: SentenceTransformer,
    datasets: dict,
    loss_dict: dict,
    config: HarnessConfig,
    eval_datasets: Optional[dict] = None,
) -> SentenceTransformerTrainer:
    """Assemble the SentenceTransformerTrainer that runs the joint hybrid objective."""
    import torch
    fp16_enabled = config.use_fp16 and torch.cuda.is_available()

    args = SentenceTransformerTrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        seed=config.seed,
        save_strategy="epoch",
        logging_steps=20,
        fp16=fp16_enabled,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    print(f"[OK] Mixed precision (fp16): {'ENABLED' if fp16_enabled else 'disabled (no GPU detected)'}")

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=datasets,
        eval_dataset=eval_datasets,
        loss=loss_dict,
    )
    print("[OK] Trainer assembled with joint contrastive + triplet objectives.")
    return trainer


def run_finetuning(
    pairs: List[dict],
    triplets: List[dict],
    config: Optional[HarnessConfig] = None,
) -> SentenceTransformer:
    """
    End-to-end entry point: load model, build hybrid Matryoshka losses,
    build datasets, train, and return the fine-tuned model.
    """
    config = config or HarnessConfig()

    model = build_model(config)
    loss_dict = build_losses(model, config)
    datasets = build_datasets(pairs, triplets)
    trainer = build_trainer(model, datasets, loss_dict, config)

    trainer.train()
    model.save(config.output_dir)
    print(f"[OK] Fine-tuned model saved to: {config.output_dir}")

    return model


if __name__ == "__main__":
    # Minimal smoke-test data so this file can be run directly to confirm the
    # harness wires together correctly. Replace with B3's placeholder data or
    # Member A's real verse-Tafsir pairs for an actual training run.
    demo_pairs = [
        {"anchor": "إِنَّ اللَّهَ غَفُورٌ رَحِيمٌ", "positive": "وَرَحْمَتِي وَسِعَتْ كُلَّ شَيْءٍ"},
        {"anchor": "وَبَشِّرِ الصَّابِرِينَ", "positive": "إِنَّ مَعَ الْعُسْرِ يُسْرًا"},
    ]
    demo_triplets = [
        {
            "anchor": "إِنَّ اللَّهَ غَفُورٌ رَحِيمٌ",
            "positive": "وَرَحْمَتِي وَسِعَتْ كُلَّ شَيْءٍ",
            "negative": "الشَّمْسُ سَاطِعَةٌ الْيَوْمَ",
        },
        {
            "anchor": "وَبَشِّرِ الصَّابِرِينَ",
            "positive": "إِنَّ مَعَ الْعُسْرِ يُسْرًا",
            "negative": "الشَّمْسُ سَاطِعَةٌ الْيَوْمَ",
        },
    ]

    cfg = HarnessConfig(num_train_epochs=1, per_device_train_batch_size=2)
    run_finetuning(demo_pairs, demo_triplets, cfg)
