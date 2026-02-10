"""
Multi-Modal Lattice Autoencoder (MMLA)
=======================================

Architecture:
  Input (any modality) → Encoder → 96-bit braille lattice → Decoder → Output

Phase 1: Single-modal (continuous vectors → lattice → reconstructed vectors)
  - Validates that the learned lattice preserves semantic structure
  - Trains on consensus corpus (126 passages, 37 traditions)
  - Classification head predicts tradition from lattice code

Phase 2: Multi-modal (text + image → shared lattice → cross-modal decode)
  - Each modality gets its own encoder, all project to same lattice
  - Cross-modal alignment: same concept from different modalities → same lattice point
  - Hamming-mean consensus across modalities

Phase 3: Operator layer
  - Fusion/fission/perturbation as differentiable lattice operations
  - Operator prediction head: given two lattice points, predict which operator applies
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lattice import (
    BrailleLatticeLayer,
    LatticeDecoder,
    DOMAINS,
    lattice_fusion,
    lattice_perturbation,
)


# ─── Phase 1: Single-Modal Lattice Autoencoder ───────────────────────

class ContinuousEncoder(nn.Module):
    """
    Encode a continuous D-dimensional vector for lattice projection.

    Maps input to a richer representation before the lattice layer
    quantizes it. This learns which features matter for each bit.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # Normalize to [0, 1] for lattice projection
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContinuousDecoder(nn.Module):
    """
    Decode from lattice bits back to continuous D-dimensional vector.

    Learns the inverse mapping from discrete lattice codes to
    continuous semantic vectors.
    """

    def __init__(self, n_bits: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TraditionClassifier(nn.Module):
    """
    Classify tradition from lattice code.

    If the lattice preserves theological structure, this should achieve
    high accuracy — proving the bottleneck retains discriminative info.
    """

    def __init__(self, n_bits: int, n_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LatticeAutoencoder(nn.Module):
    """
    Phase 1: Single-modal lattice autoencoder.

    continuous vector → encoder → lattice projection → decoder → reconstruction
                                       ↓
                                  classifier → tradition label

    Loss = reconstruction_loss + λ_cls * classification_loss + λ_commit * commitment_loss
    """

    def __init__(
        self,
        domain: str = "theology",
        hidden_dim: int = 128,
        n_classes: int = 37,
        tau_init: float = 1.0,
    ):
        super().__init__()
        domain_cfg = DOMAINS[domain]
        self.axes = domain_cfg["axes"]
        self.n_axes = len(self.axes)
        self.n_bits = self.n_axes * 8
        self.domain_name = domain

        # Encoder: continuous → prepared for lattice
        self.encoder = ContinuousEncoder(self.n_axes, hidden_dim)

        # Lattice: continuous → discrete (straight-through)
        self.lattice = BrailleLatticeLayer(
            n_axes=self.n_axes,
            polarity_pairs=domain_cfg["polarity_pairs"],
            tau_init=tau_init,
        )
        self.lattice.set_polarity_indices(self.axes)

        # Decoder: discrete → continuous reconstruction
        self.decoder = ContinuousDecoder(self.n_bits, self.n_axes, hidden_dim)

        # Classifier: discrete → tradition label
        self.classifier = TraditionClassifier(self.n_bits, n_classes)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, D) continuous belief vectors
            labels: (batch,) tradition labels (optional, for classification loss)

        Returns:
            dict with keys:
                z: (batch, D*8) binary lattice codes
                x_recon: (batch, D) reconstructed vectors
                logits: (batch, n_classes) classification logits
                loss: scalar total loss (if labels provided)
                losses: dict of individual loss components
        """
        # Encode
        h = self.encoder(x)  # (batch, D) in [0, 1]

        # Project to lattice
        z = self.lattice(h)  # (batch, D*8) binary

        # Decode
        x_recon = self.decoder(z)  # (batch, D)

        # Classify
        logits = self.classifier(z)  # (batch, n_classes)

        result = {
            "z": z,
            "x_recon": x_recon,
            "logits": logits,
            "h": h,
        }

        if labels is not None:
            # Reconstruction loss (cosine similarity)
            cos_sim = F.cosine_similarity(x, x_recon, dim=-1)
            recon_loss = (1 - cos_sim).mean()

            # L2 reconstruction loss
            l2_loss = F.mse_loss(x_recon, x)

            # Classification loss
            cls_loss = F.cross_entropy(logits, labels)

            # Commitment loss: encourage encoder output to be close to
            # what the lattice would produce from a clean projection
            # (regularizes the encoder to produce lattice-friendly representations)
            commitment_loss = F.mse_loss(h, x.detach())

            # Total loss
            total_loss = recon_loss + 0.5 * l2_loss + 0.3 * cls_loss + 0.1 * commitment_loss

            result["loss"] = total_loss
            result["losses"] = {
                "recon_cosine": recon_loss.item(),
                "recon_l2": l2_loss.item(),
                "classification": cls_loss.item(),
                "commitment": commitment_loss.item(),
                "total": total_loss.item(),
            }

        return result

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to lattice code only."""
        h = self.encoder(x)
        return self.lattice(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from lattice code only."""
        return self.decoder(z)

    def to_braille(self, x: torch.Tensor) -> list:
        """Encode and return Unicode braille strings."""
        z = self.encode(x)
        return self.lattice.to_unicode(z)


# ─── Phase 2: Multi-Modal Lattice Autoencoder ────────────────────────

class TextEncoder(nn.Module):
    """
    Encode text (as pre-computed embeddings) to lattice-ready representation.

    In Phase 2, this would take raw text and use a frozen text backbone
    (e.g., sentence-transformers) to produce embeddings, then project
    to the lattice dimension.
    """

    def __init__(self, embed_dim: int, n_axes: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_axes),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageEncoder(nn.Module):
    """
    Encode image (as pre-computed embeddings) to lattice-ready representation.

    In Phase 2, this would take raw images and use a frozen vision backbone
    (e.g., CLIP ViT) to produce embeddings, then project to lattice dimension.
    """

    def __init__(self, embed_dim: int, n_axes: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_axes),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalLatticeAutoencoder(nn.Module):
    """
    Phase 2: Multi-modal lattice autoencoder.

    Multiple modality-specific encoders project to the SAME lattice space.
    A shared decoder reconstructs from the lattice code.
    Cross-modal alignment is enforced by requiring that paired inputs
    (e.g., an image and its caption) produce the same lattice code.

    Loss = Σ_m recon_loss_m + λ_align * alignment_loss + λ_cls * cls_loss
    """

    def __init__(
        self,
        domain: str = "theology",
        text_embed_dim: int = 768,
        image_embed_dim: int = 512,
        hidden_dim: int = 256,
        n_classes: int = 37,
        tau_init: float = 1.0,
    ):
        super().__init__()
        domain_cfg = DOMAINS[domain]
        self.axes = domain_cfg["axes"]
        self.n_axes = len(self.axes)
        self.n_bits = self.n_axes * 8

        # Modality-specific encoders
        self.text_encoder = TextEncoder(text_embed_dim, self.n_axes, hidden_dim)
        self.image_encoder = ImageEncoder(image_embed_dim, self.n_axes, hidden_dim)

        # Shared lattice layer
        self.lattice = BrailleLatticeLayer(
            n_axes=self.n_axes,
            polarity_pairs=domain_cfg["polarity_pairs"],
            tau_init=tau_init,
        )
        self.lattice.set_polarity_indices(self.axes)

        # Shared decoder (reconstructs to continuous vector, not raw modality)
        self.decoder = ContinuousDecoder(self.n_bits, self.n_axes, hidden_dim)

        # Shared classifier
        self.classifier = TraditionClassifier(self.n_bits, n_classes)

    def forward(
        self,
        text_embed: Optional[torch.Tensor] = None,
        image_embed: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with one or both modalities.

        When both are provided, alignment loss encourages same lattice code.
        """
        results = {}
        z_list = []

        if text_embed is not None:
            h_text = self.text_encoder(text_embed)
            z_text = self.lattice(h_text)
            x_recon_text = self.decoder(z_text)
            results["z_text"] = z_text
            results["x_recon_text"] = x_recon_text
            results["logits_text"] = self.classifier(z_text)
            z_list.append(z_text)

        if image_embed is not None:
            h_image = self.image_encoder(image_embed)
            z_image = self.lattice(h_image)
            x_recon_image = self.decoder(z_image)
            results["z_image"] = z_image
            results["x_recon_image"] = x_recon_image
            results["logits_image"] = self.classifier(z_image)
            z_list.append(z_image)

        # Multi-modal consensus via Hamming mean
        if len(z_list) > 1:
            z_stack = torch.stack(z_list, dim=0)  # (n_modalities, batch, bits)
            z_consensus = self.lattice.hamming_centroid(
                z_stack.view(-1, self.n_bits)
            ).unsqueeze(0).expand_as(z_list[0])
            results["z_consensus"] = z_consensus

        # Compute losses if labels provided
        if labels is not None:
            losses = {}
            total = torch.tensor(0.0, device=labels.device)

            # Per-modality reconstruction + classification
            for key in ["text", "image"]:
                if f"z_{key}" in results:
                    # Reconstruction (we reconstruct to the shared semantic space)
                    # In Phase 2, the "ground truth" is the LLM-scored vector
                    pass  # Reconstruction target depends on paired data

                    # Classification
                    cls_loss = F.cross_entropy(results[f"logits_{key}"], labels)
                    losses[f"cls_{key}"] = cls_loss.item()
                    total = total + 0.3 * cls_loss

            # Cross-modal alignment: Hamming distance between modality codes
            if "z_text" in results and "z_image" in results:
                align_loss = (results["z_text"] - results["z_image"]).abs().mean()
                losses["alignment"] = align_loss.item()
                total = total + 1.0 * align_loss

            results["loss"] = total
            results["losses"] = losses

        return results

    def encode_text(self, text_embed: torch.Tensor) -> torch.Tensor:
        h = self.text_encoder(text_embed)
        return self.lattice(h)

    def encode_image(self, image_embed: torch.Tensor) -> torch.Tensor:
        h = self.image_encoder(image_embed)
        return self.lattice(h)

    def fuse(self, z_a: torch.Tensor, z_b: torch.Tensor,
             weight_a: float = 0.5) -> torch.Tensor:
        """Apply fusion operator on two lattice codes."""
        return lattice_fusion(z_a, z_b, weight_a, 1.0 - weight_a)

    def perturb(self, z: torch.Tensor, prophet: torch.Tensor,
                strength: float = 0.3) -> torch.Tensor:
        """Apply perturbation operator."""
        return lattice_perturbation(z, prophet, strength)
