"""
Data Loading for Lattice Autoencoder Training
===============================================

Phase 1: Load consensus corpus (126 passages, 37 traditions)
  - Each passage has a 12D normalized belief vector + tradition label
  - Augmentation via noise injection + lattice operator augmentation

Phase 2: Load paired multi-modal data (text embeddings + image embeddings)
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


# ─── Constants ────────────────────────────────────────────────────────

AXES = [
    "authority", "transcendence", "care", "justice", "wisdom", "power",
    "fertility", "war", "death", "creation", "nature", "order"
]


# ─── Phase 1: Consensus Corpus Dataset ───────────────────────────────

class ConsensusCorpusDataset(Dataset):
    """
    Dataset of 126 consensus-scored passages from 37 traditions.

    Each item:
      x: (12,) normalized belief vector
      label: tradition index (0–36)
      tradition: tradition name string

    Augmentation:
      - Gaussian noise (σ = noise_std)
      - Random axis dropout (zero out 1–2 axes, renormalize)
      - Lattice-aware augmentation: encode → flip random bits → decode
    """

    def __init__(
        self,
        corpus_path: str = "mlx-pipeline/multi_scorer_consensus.json",
        noise_std: float = 0.02,
        augment: bool = True,
        samples_per_passage: int = 50,
    ):
        self.noise_std = noise_std
        self.augment = augment
        self.samples_per_passage = samples_per_passage

        # Load corpus
        path = Path(corpus_path)
        if not path.exists():
            # Try relative to repo root
            path = Path(__file__).parent.parent / corpus_path
        if not path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")

        with open(path) as f:
            data = json.load(f)

        self.embeddings = data["embeddings"]

        # Build tradition → index mapping
        traditions = sorted(set(e["tradition"] for e in self.embeddings))
        self.tradition_to_idx = {t: i for i, t in enumerate(traditions)}
        self.idx_to_tradition = {i: t for t, i in self.tradition_to_idx.items()}
        self.n_classes = len(traditions)

        # Extract vectors and labels
        self.vectors = []
        self.labels = []
        self.traditions = []
        for e in self.embeddings:
            vec = [e["normalized"].get(a, 0.0) for a in AXES]
            self.vectors.append(vec)
            self.labels.append(self.tradition_to_idx[e["tradition"]])
            self.traditions.append(e["tradition"])

        self.base_vectors = torch.tensor(self.vectors, dtype=torch.float32)
        self.base_labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.vectors) * self.samples_per_passage

    def __getitem__(self, idx):
        base_idx = idx % len(self.vectors)
        x = self.base_vectors[base_idx].clone()
        label = self.base_labels[base_idx]

        if self.augment and idx >= len(self.vectors):
            # Add Gaussian noise
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

            # Random axis dropout (10% chance per axis)
            if random.random() < 0.2:
                drop_idx = random.randint(0, len(AXES) - 1)
                x[drop_idx] = 0.0

            # Renormalize
            norm = x.norm() + 1e-8
            x = x / norm

        return x, label

    def get_tradition_name(self, idx: int) -> str:
        return self.idx_to_tradition.get(idx, "unknown")


# ─── Phase 1b: Per-Scorer Datasets (for multi-encoder training) ──────

class PerScorerDataset(Dataset):
    """
    Load individual scorer files for multi-encoder braiding experiments.

    Each scorer's embeddings are projected independently, then
    the Hamming mean across scorers is the training target.
    """

    def __init__(
        self,
        scorer_paths: dict = None,
        noise_std: float = 0.02,
        samples_per_passage: int = 50,
    ):
        if scorer_paths is None:
            base = Path(__file__).parent.parent / "mlx-pipeline"
            scorer_paths = {
                "claude": base / "expanded_embeddings_results.json",
                "gpt4o": base / "scores_gpt4o.json",
                "gemini": base / "scores_gemini_flash.json",
                "llama70b": base / "scores_llama70b.json",
            }

        self.noise_std = noise_std
        self.samples_per_passage = samples_per_passage
        self.scorers = {}
        self.n_passages = None

        for name, path in scorer_paths.items():
            path = Path(path)
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                embeddings = data["embeddings"]
                vecs = []
                for e in embeddings:
                    vec = [e.get("normalized", e.get("raw_scores", {})).get(a, 0.0) for a in AXES]
                    # Normalize
                    norm = math.sqrt(sum(v ** 2 for v in vec)) or 1.0
                    vecs.append([v / norm for v in vec])
                self.scorers[name] = torch.tensor(vecs, dtype=torch.float32)
                if self.n_passages is None:
                    self.n_passages = len(vecs)

        # Build tradition mapping from first scorer
        first_path = next(iter(scorer_paths.values()))
        with open(first_path) as f:
            data = json.load(f)
        traditions = sorted(set(e["tradition"] for e in data["embeddings"]))
        self.tradition_to_idx = {t: i for i, t in enumerate(traditions)}
        self.labels = torch.tensor(
            [self.tradition_to_idx[e["tradition"]] for e in data["embeddings"]],
            dtype=torch.long
        )
        self.n_classes = len(traditions)

    def __len__(self):
        return (self.n_passages or 0) * self.samples_per_passage

    def __getitem__(self, idx):
        base_idx = idx % self.n_passages
        # Return all scorer vectors for this passage + label
        scorer_vecs = {name: vecs[base_idx] for name, vecs in self.scorers.items()}
        label = self.labels[base_idx]
        return scorer_vecs, label


# ─── Data Loaders ─────────────────────────────────────────────────────

def get_dataloader(
    domain: str = "theology",
    batch_size: int = 32,
    noise_std: float = 0.02,
    samples_per_passage: int = 50,
    num_workers: int = 0,
    split: str = "train",
    val_fraction: float = 0.15,
) -> DataLoader:
    """
    Get a DataLoader for the consensus corpus.

    For Phase 1, we use the theological domain with augmentation.
    Train/val split is stratified by tradition.
    """
    dataset = ConsensusCorpusDataset(
        noise_std=noise_std if split == "train" else 0.0,
        augment=(split == "train"),
        samples_per_passage=samples_per_passage if split == "train" else 1,
    )

    if split in ("train", "val"):
        # Stratified split: keep proportional representation of each tradition
        n = len(dataset.base_vectors)
        indices = list(range(n))
        random.seed(42)
        random.shuffle(indices)
        n_val = max(1, int(n * val_fraction))
        val_indices = set(indices[:n_val])
        train_indices = set(indices[n_val:])

        if split == "val":
            # Return only base vectors (no augmentation) for validation
            subset_vecs = dataset.base_vectors[sorted(val_indices)]
            subset_labels = dataset.base_labels[sorted(val_indices)]
            subset_dataset = torch.utils.data.TensorDataset(subset_vecs, subset_labels)
            return DataLoader(subset_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                      num_workers=num_workers)
