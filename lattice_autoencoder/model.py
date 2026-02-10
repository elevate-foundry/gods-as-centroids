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
    Encode text embeddings (pre-computed TF-IDF+PCA) to lattice-ready D-dim.
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


class ScorerEncoder(nn.Module):
    """
    Encode an individual LLM scorer's 12D theological vector to lattice-ready D-dim.
    This is the "second modality" — a different model's view of the same passage.
    """

    def __init__(self, n_axes: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_axes, hidden_dim),
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

    Two modality-specific encoders project to the SAME lattice space:
      1. TextEncoder: text embeddings (126-dim TF-IDF+PCA) → 12D → lattice
      2. ScorerEncoder: individual LLM scorer vectors (12D) → 12D → lattice

    Both must produce the same lattice code for the same passage.
    The shared decoder reconstructs the consensus theological vector from
    the lattice code.

    Loss = recon_text + recon_scorer + λ_align * alignment + λ_cls * classification
    """

    def __init__(
        self,
        domain: str = "theology",
        text_embed_dim: int = 126,
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
        self.scorer_encoder = ScorerEncoder(self.n_axes, hidden_dim // 2)

        # Shared lattice layer
        self.lattice = BrailleLatticeLayer(
            n_axes=self.n_axes,
            polarity_pairs=domain_cfg["polarity_pairs"],
            tau_init=tau_init,
        )
        self.lattice.set_polarity_indices(self.axes)

        # Shared decoder: lattice → consensus theological vector
        self.decoder = ContinuousDecoder(self.n_bits, self.n_axes, hidden_dim)

        # Shared classifier
        self.classifier = TraditionClassifier(self.n_bits, n_classes)

    def forward(
        self,
        text_embed: Optional[torch.Tensor] = None,
        scorer_vec: Optional[torch.Tensor] = None,
        consensus_target: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with one or both modalities.

        Args:
            text_embed: (batch, text_dim) text embeddings
            scorer_vec: (batch, 12) individual scorer theological vector
            consensus_target: (batch, 12) consensus vector (reconstruction target)
            labels: (batch,) tradition labels
        """
        results = {}

        if text_embed is not None:
            h_text = self.text_encoder(text_embed)
            z_text = self.lattice(h_text)
            recon_text = self.decoder(z_text)
            logits_text = self.classifier(z_text)
            results["z_text"] = z_text
            results["recon_text"] = recon_text
            results["logits_text"] = logits_text

        if scorer_vec is not None:
            h_scorer = self.scorer_encoder(scorer_vec)
            z_scorer = self.lattice(h_scorer)
            recon_scorer = self.decoder(z_scorer)
            logits_scorer = self.classifier(z_scorer)
            results["z_scorer"] = z_scorer
            results["recon_scorer"] = recon_scorer
            results["logits_scorer"] = logits_scorer

        # Multi-modal consensus via Hamming mean (per-sample)
        if "z_text" in results and "z_scorer" in results:
            # Per-sample majority vote across 2 modalities
            z_mean = (results["z_text"] + results["z_scorer"]) / 2.0
            z_consensus = (z_mean > 0.5).float()
            results["z_consensus"] = z_consensus
            results["recon_consensus"] = self.decoder(z_consensus)
            results["logits_consensus"] = self.classifier(z_consensus)

        # Compute losses
        if consensus_target is not None and labels is not None:
            losses = {}
            total = torch.tensor(0.0, device=labels.device, requires_grad=True)

            # Per-modality reconstruction loss (target = consensus vector)
            for key in ["text", "scorer"]:
                if f"recon_{key}" in results:
                    cos_loss = (1 - F.cosine_similarity(
                        consensus_target, results[f"recon_{key}"], dim=-1
                    )).mean()
                    l2_loss = F.mse_loss(results[f"recon_{key}"], consensus_target)
                    losses[f"recon_{key}_cos"] = cos_loss.item()
                    losses[f"recon_{key}_l2"] = l2_loss.item()
                    total = total + cos_loss + 0.5 * l2_loss

            # Per-modality classification loss
            for key in ["text", "scorer"]:
                if f"logits_{key}" in results:
                    cls_loss = F.cross_entropy(results[f"logits_{key}"], labels)
                    losses[f"cls_{key}"] = cls_loss.item()
                    total = total + 0.3 * cls_loss

            # Cross-modal alignment: encourage same lattice code
            if "z_text" in results and "z_scorer" in results:
                # Bit-level agreement loss (differentiable via soft codes)
                align_loss = F.mse_loss(results["z_text"], results["z_scorer"])
                losses["alignment"] = align_loss.item()
                total = total + 2.0 * align_loss

            # Consensus reconstruction bonus
            if "recon_consensus" in results:
                cons_cos = (1 - F.cosine_similarity(
                    consensus_target, results["recon_consensus"], dim=-1
                )).mean()
                losses["recon_consensus"] = cons_cos.item()
                total = total + 0.5 * cons_cos

            results["loss"] = total
            results["losses"] = losses

        return results

    def encode_text(self, text_embed: torch.Tensor) -> torch.Tensor:
        h = self.text_encoder(text_embed)
        return self.lattice(h)

    def encode_scorer(self, scorer_vec: torch.Tensor) -> torch.Tensor:
        h = self.scorer_encoder(scorer_vec)
        return self.lattice(h)

    def fuse(self, z_a: torch.Tensor, z_b: torch.Tensor,
             weight_a: float = 0.5) -> torch.Tensor:
        """Apply fusion operator on two lattice codes."""
        return lattice_fusion(z_a, z_b, weight_a, 1.0 - weight_a)

    def perturb(self, z: torch.Tensor, prophet: torch.Tensor,
                strength: float = 0.3) -> torch.Tensor:
        """Apply perturbation operator."""
        return lattice_perturbation(z, prophet, strength)


# ─── Phase 3: Operator Layer ─────────────────────────────────────────

class DifferentiableOperatorLayer(nn.Module):
    """
    Phase 3: Differentiable lattice operators with gradient flow.

    Three operators on lattice codes:
      1. Fusion: merge two lattice points → one (syncretism)
      2. Fission: split one lattice point → two (schism)
      3. Perturbation: shift a lattice point toward a prophet vector

    Each operator is parameterized by a learned weight network that
    determines how strongly each bit is influenced.
    """

    def __init__(self, n_bits: int, hidden_dim: int = 64):
        super().__init__()
        self.n_bits = n_bits

        # Fusion: learns per-bit weighting for merging two codes
        self.fusion_net = nn.Sequential(
            nn.Linear(n_bits * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_bits),
            nn.Sigmoid(),  # Per-bit weight: how much to take from A vs B
        )

        # Fission: learns a split direction from a single code
        self.fission_net = nn.Sequential(
            nn.Linear(n_bits, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_bits),
            nn.Tanh(),  # Per-bit perturbation direction
        )

        # Perturbation: learns pull strength per bit
        self.perturbation_net = nn.Sequential(
            nn.Linear(n_bits * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_bits),
            nn.Sigmoid(),  # Per-bit pull strength
        )

    def fusion(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Learned fusion: merge two lattice codes.
        Output is a soft blend, binarized via straight-through.
        """
        combined = torch.cat([z_a, z_b], dim=-1)
        weights = self.fusion_net(combined)  # (batch, n_bits) in [0, 1]
        soft = weights * z_a + (1 - weights) * z_b
        # Straight-through binarize
        hard = (soft > 0.5).float()
        return hard + soft - soft.detach()  # STE

    def fission(self, z: torch.Tensor) -> tuple:
        """
        Learned fission: split one lattice code into two.
        Returns (z_a, z_b) — two child codes.
        """
        perturbation = self.fission_net(z)  # (batch, n_bits) in [-1, 1]
        # Child A: flip bits where perturbation > 0
        soft_a = z + 0.3 * perturbation
        soft_b = z - 0.3 * perturbation
        hard_a = (soft_a > 0.5).float()
        hard_b = (soft_b > 0.5).float()
        # STE
        z_a = hard_a + soft_a - soft_a.detach()
        z_b = hard_b + soft_b - soft_b.detach()
        return z_a, z_b

    def perturbation(self, z_centroid: torch.Tensor,
                     z_prophet: torch.Tensor) -> torch.Tensor:
        """
        Learned perturbation: shift centroid toward prophet.
        """
        combined = torch.cat([z_centroid, z_prophet], dim=-1)
        pull = self.perturbation_net(combined)  # per-bit pull strength
        soft = (1 - pull) * z_centroid + pull * z_prophet
        hard = (soft > 0.5).float()
        return hard + soft - soft.detach()  # STE


class OperatorPredictionHead(nn.Module):
    """
    Given two lattice codes, predict which operator should be applied.

    Classes:
      0 = fusion (codes are similar, should merge)
      1 = fission (codes are distant, should split)
      2 = perturbation (one is a "prophet" — high-magnitude outlier)
      3 = none (codes are stable, no operation needed)
    """

    def __init__(self, n_bits: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits * 2 + 4, hidden_dim),  # +4 for distance features
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),  # 4 operator classes
        )

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Predict operator class from two lattice codes."""
        # Compute distance features
        hamming = (z_a != z_b).float().sum(dim=-1, keepdim=True)
        hamming_norm = hamming / z_a.shape[-1]
        bit_and = (z_a * z_b).sum(dim=-1, keepdim=True) / z_a.shape[-1]
        bit_xor = ((z_a + z_b) % 2).sum(dim=-1, keepdim=True) / z_a.shape[-1]

        features = torch.cat([
            z_a, z_b,
            hamming_norm, bit_and, bit_xor,
            hamming / z_a.shape[-1],  # redundant but helps gradient flow
        ], dim=-1)

        return self.net(features)


class LatticeOperatorModel(nn.Module):
    """
    Phase 3: Complete operator model.

    Combines:
      1. A trained Phase 1 or Phase 2 encoder (frozen or fine-tuned)
      2. Differentiable operator layer
      3. Operator prediction head
      4. Shared decoder for reconstruction validation

    Training:
      - Generate operator examples from the corpus:
        * Fusion: pairs of same-family traditions
        * Fission: traditions with known historical schisms
        * Perturbation: tradition + outlier passage
        * None: same tradition, similar passages
      - Train operator prediction + operator execution jointly
    """

    def __init__(
        self,
        n_bits: int = 96,
        n_axes: int = 12,
        n_classes: int = 37,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.n_axes = n_axes

        # Operator layer
        self.operators = DifferentiableOperatorLayer(n_bits, hidden_dim // 2)

        # Operator prediction head
        self.predictor = OperatorPredictionHead(n_bits, hidden_dim // 2)

        # Decoder for validating operator outputs
        self.decoder = ContinuousDecoder(n_bits, n_axes, hidden_dim)

        # Classifier for validating operator outputs preserve tradition
        self.classifier = TraditionClassifier(n_bits, n_classes)

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        op_labels: Optional[torch.Tensor] = None,
        target_vecs: Optional[torch.Tensor] = None,
        target_labels: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            z_a: (batch, n_bits) first lattice code
            z_b: (batch, n_bits) second lattice code
            op_labels: (batch,) ground truth operator class (0=fusion, 1=fission, 2=perturb, 3=none)
            target_vecs: (batch, n_axes) expected output continuous vector
            target_labels: (batch,) expected output tradition label
        """
        # Predict operator
        op_logits = self.predictor(z_a, z_b)

        # Apply all operators (we'll select based on prediction during inference)
        z_fused = self.operators.fusion(z_a, z_b)
        z_fiss_a, z_fiss_b = self.operators.fission(z_a)
        z_perturbed = self.operators.perturbation(z_a, z_b)

        results = {
            "op_logits": op_logits,
            "z_fused": z_fused,
            "z_fiss_a": z_fiss_a,
            "z_fiss_b": z_fiss_b,
            "z_perturbed": z_perturbed,
        }

        # Decode all operator outputs
        results["recon_fused"] = self.decoder(z_fused)
        results["recon_perturbed"] = self.decoder(z_perturbed)
        results["logits_fused"] = self.classifier(z_fused)

        if op_labels is not None:
            losses = {}
            total = torch.tensor(0.0, device=op_labels.device, requires_grad=True)

            # Operator prediction loss
            pred_loss = F.cross_entropy(op_logits, op_labels)
            losses["op_prediction"] = pred_loss.item()
            total = total + pred_loss

            # For fusion examples: reconstruction should match target
            if target_vecs is not None:
                fusion_mask = (op_labels == 0).float()
                if fusion_mask.sum() > 0:
                    cos_loss = (1 - F.cosine_similarity(
                        target_vecs, results["recon_fused"], dim=-1
                    ))
                    fusion_recon = (cos_loss * fusion_mask).sum() / (fusion_mask.sum() + 1e-8)
                    losses["fusion_recon"] = fusion_recon.item()
                    total = total + fusion_recon

            # For perturbation examples: perturbed code should be closer to z_b
            perturb_mask = (op_labels == 2).float()
            if perturb_mask.sum() > 0:
                # Perturbed should be between z_a and z_b
                dist_to_b = (z_perturbed - z_b).abs().mean(dim=-1)
                dist_to_a = (z_perturbed - z_a).abs().mean(dim=-1)
                # Should be closer to b than a was
                orig_dist = (z_a - z_b).abs().mean(dim=-1)
                pull_loss = F.relu(dist_to_b - 0.7 * orig_dist)
                perturb_loss = (pull_loss * perturb_mask).sum() / (perturb_mask.sum() + 1e-8)
                losses["perturbation"] = perturb_loss.item()
                total = total + perturb_loss

            # Fission: children should be more distant than parent
            fission_mask = (op_labels == 1).float()
            if fission_mask.sum() > 0:
                child_dist = (z_fiss_a - z_fiss_b).abs().mean(dim=-1)
                # Children should be at least somewhat different
                fission_loss = F.relu(0.2 - child_dist)
                fission_loss = (fission_loss * fission_mask).sum() / (fission_mask.sum() + 1e-8)
                losses["fission"] = fission_loss.item()
                total = total + fission_loss

            # Classification: fused output should be classifiable
            if target_labels is not None:
                cls_loss = F.cross_entropy(results["logits_fused"], target_labels)
                losses["classification"] = cls_loss.item()
                total = total + 0.3 * cls_loss

            results["loss"] = total
            results["losses"] = losses

        return results

    def predict_and_apply(self, z_a: torch.Tensor, z_b: torch.Tensor):
        """Inference: predict operator and apply it."""
        op_logits = self.predictor(z_a, z_b)
        op_pred = op_logits.argmax(dim=-1)

        batch_size = z_a.shape[0]
        outputs = torch.zeros_like(z_a)

        for i in range(batch_size):
            op = op_pred[i].item()
            if op == 0:  # fusion
                outputs[i] = self.operators.fusion(z_a[i:i+1], z_b[i:i+1]).squeeze(0)
            elif op == 1:  # fission
                a, b = self.operators.fission(z_a[i:i+1])
                outputs[i] = a.squeeze(0)  # return first child
            elif op == 2:  # perturbation
                outputs[i] = self.operators.perturbation(z_a[i:i+1], z_b[i:i+1]).squeeze(0)
            else:  # none
                outputs[i] = z_a[i]

        return outputs, op_pred
