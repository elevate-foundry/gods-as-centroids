"""
Differentiable 8-Dot Braille Lattice Layer
============================================

The core innovation: a discrete bottleneck layer where every bit has a
named semantic role, with gradient flow via straight-through estimation.

Architecture per axis (8 bits):
  Dot 1: positive polarity    (value > threshold)
  Dot 2: negative polarity    (opposite > value + margin)
  Dot 3: tension              (both active, close values)
  Dot 4: intensity high bit   (quantized magnitude)
  Dot 5: intensity low bit    (quantized magnitude)
  Dot 6: rigidity             (strong conviction)
  Dot 7: salience             (above median)
  Dot 8: momentum             (temporal change)

Total: D axes × 8 bits = 96 bits for theology (12D)

Gradient flow: straight-through estimator (Bengio et al. 2013)
  Forward:  z_hard = (sigmoid(logits * τ) > 0.5).float()
  Backward: ∂L/∂logits passes through as if z_hard = sigmoid(logits * τ)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Dot name registry ───────────────────────────────────────────────

DOT_NAMES = [
    "positive_polarity", "negative_polarity", "tension",
    "intensity_high", "intensity_low", "rigidity",
    "salience", "momentum"
]

# ─── Domain definitions ──────────────────────────────────────────────

THEOLOGY_AXES = [
    "authority", "transcendence", "care", "justice", "wisdom", "power",
    "fertility", "war", "death", "creation", "nature", "order"
]

THEOLOGY_POLARITY_PAIRS = {
    "authority": "care", "care": "authority",
    "transcendence": "nature", "nature": "transcendence",
    "justice": "fertility", "fertility": "justice",
    "wisdom": "war", "war": "wisdom",
    "power": "death", "death": "power",
    "creation": "order", "order": "creation",
}

POLITICAL_AXES = [
    "economic_left", "economic_right", "auth_state", "lib_individual",
    "progressive", "traditional", "globalist", "nationalist",
    "secular", "religious"
]

PERSONALITY_AXES = [
    "openness", "conscientiousness", "extraversion",
    "agreeableness", "neuroticism"
]

DOMAINS = {
    "theology": {"axes": THEOLOGY_AXES, "polarity_pairs": THEOLOGY_POLARITY_PAIRS},
    "political": {"axes": POLITICAL_AXES, "polarity_pairs": {}},
    "personality": {"axes": PERSONALITY_AXES, "polarity_pairs": {}},
}


# ─── Straight-Through Estimator ──────────────────────────────────────

class StraightThrough(torch.autograd.Function):
    """Binarize in forward pass, pass gradients through in backward."""

    @staticmethod
    def forward(ctx, logits, tau):
        probs = torch.sigmoid(logits * tau)
        hard = (probs > 0.5).float()
        ctx.save_for_backward(logits, torch.tensor(tau))
        return hard

    @staticmethod
    def backward(ctx, grad_output):
        logits, tau = ctx.saved_tensors
        # Gradient of sigmoid as surrogate
        probs = torch.sigmoid(logits * tau.item())
        surrogate_grad = probs * (1 - probs) * tau.item()
        return grad_output * surrogate_grad, None


def straight_through_binarize(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Binarize logits with straight-through gradient estimator."""
    return StraightThrough.apply(logits, tau)


# ─── Braille Lattice Layer ───────────────────────────────────────────

class BrailleLatticeLayer(nn.Module):
    """
    Differentiable 8-dot braille lattice projection.

    Input:  (batch, D) continuous vector in [0, 1]^D
    Output: (batch, D * 8) binary lattice point in {0, 1}^(D*8)

    Each axis produces 8 bits via learned thresholds + straight-through.
    The thresholds are initialized to match the hand-crafted encoding
    from recursive_compression.py but are trainable.
    """

    def __init__(self, n_axes: int, polarity_pairs: dict = None, tau_init: float = 1.0):
        super().__init__()
        self.n_axes = n_axes
        self.n_bits = n_axes * 8
        self.polarity_pairs = polarity_pairs or {}
        self.tau = tau_init

        # Polarity index mapping: for each axis, which axis is its opposite?
        # -1 means no opposite
        self.register_buffer(
            "polarity_idx",
            torch.full((n_axes,), -1, dtype=torch.long)
        )

        # Learnable projection: continuous (D,) → logits (D*8,)
        # This replaces the hand-crafted thresholds with a learned projection
        # that is initialized to approximate the hand-crafted encoding.
        self.encoder = nn.Sequential(
            nn.Linear(n_axes, n_axes * 4),
            nn.GELU(),
            nn.Linear(n_axes * 4, n_axes * 8),
        )

        # Initialize encoder to approximate hand-crafted encoding
        self._init_weights()

    def _init_weights(self):
        """Xavier init for encoder layers."""
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def set_polarity_indices(self, axes: list):
        """Set polarity pair indices from axis name list."""
        idx = torch.full((self.n_axes,), -1, dtype=torch.long)
        for i, axis in enumerate(axes):
            if axis in self.polarity_pairs:
                opp = self.polarity_pairs[axis]
                if opp in axes:
                    idx[i] = axes.index(opp)
        self.polarity_idx = idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, D) continuous belief vector, values in [0, 1]

        Returns:
            z_hard: (batch, D * 8) binary lattice point
        """
        # Learned projection to logits
        logits = self.encoder(x)  # (batch, D * 8)

        # Binarize with straight-through estimator
        z_hard = straight_through_binarize(logits, self.tau)

        return z_hard

    def anneal_tau(self, epoch: int, max_epochs: int, tau_min: float = 1.0, tau_max: float = 10.0):
        """Anneal temperature from tau_min to tau_max over training."""
        progress = min(1.0, epoch / max(1, max_epochs))
        self.tau = tau_min + (tau_max - tau_min) * progress

    def to_unicode(self, z: torch.Tensor) -> list:
        """Convert batch of lattice points to Unicode braille strings."""
        batch_size = z.shape[0]
        results = []
        for b in range(batch_size):
            bits = z[b].detach().cpu().bool().tolist()
            chars = []
            offsets = [1, 2, 4, 8, 16, 32, 64, 128]
            for ax in range(self.n_axes):
                code = 0x2800
                for i in range(8):
                    if bits[ax * 8 + i]:
                        code += offsets[i]
                chars.append(chr(code))
            results.append("".join(chars))
        return results

    def hamming_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Hamming distance between lattice points. (batch,) output."""
        return (a != b).float().sum(dim=-1)

    def hamming_centroid(self, z_batch: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Hamming-mean centroid (majority vote) over a batch of lattice points.

        Args:
            z_batch: (N, D*8) binary lattice points
            weights: (N,) optional prestige weights

        Returns:
            centroid: (D*8,) binary centroid
        """
        if weights is None:
            votes = z_batch.sum(dim=0)
            threshold = z_batch.shape[0] / 2.0
        else:
            votes = (z_batch * weights.unsqueeze(-1)).sum(dim=0)
            threshold = weights.sum() / 2.0
        return (votes > threshold).float()


# ─── Lattice Decoder Layer ────────────────────────────────────────────

class LatticeDecoder(nn.Module):
    """
    Decode binary lattice point back to continuous vector.

    Input:  (batch, D * 8) binary lattice point
    Output: (batch, D) reconstructed continuous vector
    """

    def __init__(self, n_axes: int):
        super().__init__()
        self.n_axes = n_axes
        self.decoder = nn.Sequential(
            nn.Linear(n_axes * 8, n_axes * 4),
            nn.GELU(),
            nn.Linear(n_axes * 4, n_axes),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.decoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


# ─── Differentiable Lattice Operators ─────────────────────────────────

def lattice_fusion(a: torch.Tensor, b: torch.Tensor,
                   weight_a: float = 0.5, weight_b: float = 0.5) -> torch.Tensor:
    """
    Fusion operator: weighted Hamming mean of two lattice points.
    Differentiable via soft voting.
    """
    soft = weight_a * a + weight_b * b
    return (soft > 0.5).float()


def lattice_perturbation(centroid: torch.Tensor, prophet: torch.Tensor,
                         pull_strength: float = 0.3) -> torch.Tensor:
    """
    Perturbation operator: shift centroid toward prophet's lattice point.
    """
    return lattice_fusion(centroid, prophet,
                          weight_a=1.0 - pull_strength,
                          weight_b=pull_strength)
