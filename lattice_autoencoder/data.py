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


# ─── Phase 2: Multi-Modal Dataset ─────────────────────────────────────

class MultiModalDataset(Dataset):
    """
    Multi-modal dataset for Phase 2 training.

    Three modalities, all describing the same 126 passages:
      1. Text embeddings (TF-IDF + PCA from passage text)
      2. Scorer vectors (individual LLM theological scores — 4 "views")
      3. Consensus target (4-model mean theological vector)

    Each item returns:
      text_embed: (text_dim,) text embedding
      scorer_vec: (12,) one randomly-selected scorer's theological vector
      consensus_vec: (12,) consensus theological vector (training target)
      label: tradition index
    """

    def __init__(
        self,
        text_embed_path: str = "lattice_autoencoder/data/text_embeddings.json",
        consensus_path: str = "mlx-pipeline/multi_scorer_consensus.json",
        scorer_paths: dict = None,
        noise_std: float = 0.02,
        augment: bool = True,
        samples_per_passage: int = 50,
    ):
        self.noise_std = noise_std
        self.augment = augment
        self.samples_per_passage = samples_per_passage

        root = Path(__file__).parent.parent

        # Load text embeddings
        text_path = Path(text_embed_path)
        if not text_path.exists():
            text_path = root / text_embed_path
        with open(text_path) as f:
            text_data = json.load(f)
        self.text_dim = text_data["embed_dim"]
        self.text_embeds = torch.tensor(
            [e["vector"] for e in text_data["embeddings"]], dtype=torch.float32
        )

        # Load consensus scores
        cons_path = Path(consensus_path)
        if not cons_path.exists():
            cons_path = root / consensus_path
        with open(cons_path) as f:
            cons_data = json.load(f)

        traditions = sorted(set(e["tradition"] for e in cons_data["embeddings"]))
        self.tradition_to_idx = {t: i for i, t in enumerate(traditions)}
        self.idx_to_tradition = {i: t for t, i in self.tradition_to_idx.items()}
        self.n_classes = len(traditions)

        self.consensus_vecs = []
        self.labels = []
        self.traditions_list = []
        for e in cons_data["embeddings"]:
            vec = [e["normalized"].get(a, 0.0) for a in AXES]
            self.consensus_vecs.append(vec)
            self.labels.append(self.tradition_to_idx[e["tradition"]])
            self.traditions_list.append(e["tradition"])

        self.consensus_vecs = torch.tensor(self.consensus_vecs, dtype=torch.float32)
        self.labels_tensor = torch.tensor(self.labels, dtype=torch.long)

        # Load per-scorer vectors
        if scorer_paths is None:
            scorer_paths = {
                "claude": root / "mlx-pipeline" / "expanded_embeddings_results.json",
                "gpt4o": root / "mlx-pipeline" / "scores_gpt4o.json",
                "gemini": root / "mlx-pipeline" / "scores_gemini_flash.json",
                "llama70b": root / "mlx-pipeline" / "scores_llama70b.json",
            }

        self.scorer_vecs = []  # list of (n_passages, 12) tensors
        self.scorer_names = []
        for name, path in scorer_paths.items():
            path = Path(path)
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                vecs = []
                for e in data["embeddings"]:
                    scores = e.get("normalized", e.get("raw_scores", {}))
                    vec = [scores.get(a, 0.0) for a in AXES]
                    norm = math.sqrt(sum(v ** 2 for v in vec)) or 1.0
                    vecs.append([v / norm for v in vec])
                self.scorer_vecs.append(torch.tensor(vecs, dtype=torch.float32))
                self.scorer_names.append(name)

        self.n_scorers = len(self.scorer_vecs)
        self.n_passages = len(self.consensus_vecs)

    def __len__(self):
        return self.n_passages * self.samples_per_passage

    def __getitem__(self, idx):
        base_idx = idx % self.n_passages

        # Text embedding
        text_embed = self.text_embeds[base_idx].clone()

        # Randomly select one scorer's vector as the "second modality"
        scorer_idx = random.randint(0, self.n_scorers - 1)
        scorer_vec = self.scorer_vecs[scorer_idx][base_idx].clone()

        # Consensus target
        consensus_vec = self.consensus_vecs[base_idx].clone()

        label = self.labels_tensor[base_idx]

        if self.augment and idx >= self.n_passages:
            # Add noise to text embedding
            text_embed = text_embed + torch.randn_like(text_embed) * self.noise_std
            text_embed = text_embed / (text_embed.norm() + 1e-8)

            # Add noise to scorer vector
            scorer_vec = scorer_vec + torch.randn_like(scorer_vec) * self.noise_std
            scorer_vec = scorer_vec / (scorer_vec.norm() + 1e-8)

        return text_embed, scorer_vec, consensus_vec, label


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
