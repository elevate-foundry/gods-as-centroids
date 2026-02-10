#!/usr/bin/env python3
"""
Recursive Semantic Compression Engine
======================================
The AGI blueprint made concrete. Three layers of compression:

  Layer 1: Continuous belief vectors (ℝ^D)
  Layer 2: 8-dot Braille lattice projection ({0,1}^(D×8))
  Layer 3: Hamming-mean centroids (snap dynamics)

Plus a composable operator layer:
  - Fusion:       merge two lattice attractors
  - Fission:      split one attractor into two
  - Perturbation: shift an attractor by a high-prestige agent

This module is domain-agnostic: it works on theology (12D), political
ideology (10D), personality (5D), or any continuous embedding space.

Usage:
  python sim/recursive_compression.py          # run full demo
  python sim/recursive_compression.py --test   # run validation suite
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════
# §1  DOMAIN DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

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


@dataclass
class Domain:
    """A semantic domain: axes + optional polarity pairs."""
    name: str
    axes: List[str]
    polarity_pairs: Dict[str, str] = field(default_factory=dict)

    @property
    def D(self) -> int:
        return len(self.axes)

    @property
    def bits_per_axis(self) -> int:
        return 8  # 8-dot braille

    @property
    def total_bits(self) -> int:
        return self.D * self.bits_per_axis


THEOLOGY = Domain("theology", THEOLOGY_AXES, THEOLOGY_POLARITY_PAIRS)
POLITICAL = Domain("political", POLITICAL_AXES)
PERSONALITY = Domain("personality", PERSONALITY_AXES)


# ═══════════════════════════════════════════════════════════════════════
# §2  CONTINUOUS VECTOR OPERATIONS (Layer 1)
# ═══════════════════════════════════════════════════════════════════════

Vec = Dict[str, float]


def zero_vec(axes: List[str]) -> Vec:
    return {a: 0.0 for a in axes}


def rand_vec(axes: List[str], rng: random.Random) -> Vec:
    v = {a: rng.random() for a in axes}
    return normalize(v, axes)


def normalize(v: Vec, axes: List[str]) -> Vec:
    n = math.sqrt(sum(v[a] ** 2 for a in axes)) or 1.0
    return {a: v[a] / n for a in axes}


def cosine_sim(a: Vec, b: Vec, axes: List[str]) -> float:
    dot = sum(a[k] * b[k] for k in axes)
    na = math.sqrt(sum(a[k] ** 2 for k in axes)) or 1.0
    nb = math.sqrt(sum(b[k] ** 2 for k in axes)) or 1.0
    return dot / (na * nb)


def vec_add_scaled(target: Vec, source: Vec, scale: float, axes: List[str]) -> None:
    for k in axes:
        target[k] += source[k] * scale


def arithmetic_mean(vecs: List[Vec], axes: List[str], weights: Optional[List[float]] = None) -> Vec:
    """Standard arithmetic mean centroid (Layer 1 — continuous)."""
    if not vecs:
        return zero_vec(axes)
    result = zero_vec(axes)
    if weights is None:
        for v in vecs:
            vec_add_scaled(result, v, 1.0, axes)
        return normalize(result, axes)
    total_w = sum(weights)
    for v, w in zip(vecs, weights):
        vec_add_scaled(result, v, w, axes)
    return normalize(result, axes) if total_w > 0 else zero_vec(axes)


# ═══════════════════════════════════════════════════════════════════════
# §3  8-DOT BRAILLE LATTICE (Layer 2)
# ═══════════════════════════════════════════════════════════════════════
#
# Extension from 6-dot (72-bit) to 8-dot (96-bit for 12 axes):
#
#   Standard 6-dot cell:        Extended 8-dot cell:
#     1 4                         1 4
#     2 5                         2 5
#     3 6                         3 6
#                                 7 8   ← NEW
#
#   Dot semantics per axis:
#     Dots 1-3: polarity (positive, negative, tension)
#     Dots 4-5: intensity (4 levels: 00, 01, 10, 11)
#     Dot 6:    rigidity (fluid vs dogmatic)
#     Dot 7:    salience (is this axis contextually active?)     ← NEW
#     Dot 8:    momentum (is this axis currently changing?)      ← NEW
#
#   Total: D axes × 8 dots = D×8 bits
#   Theology: 12 × 8 = 96 bits
#   Political: 10 × 8 = 80 bits
#   Personality: 5 × 8 = 40 bits

@dataclass
class BrailleCell:
    """8-dot braille cell encoding one semantic axis."""
    dots: List[bool]  # exactly 8 booleans

    def to_unicode(self) -> str:
        """Render as Unicode braille character (8-dot, U+2800-U+28FF)."""
        code = 0x2800
        offsets = [1, 2, 4, 8, 16, 32, 64, 128]
        for i, d in enumerate(self.dots):
            if d:
                code += offsets[i]
        return chr(code)

    def to_bits(self) -> List[bool]:
        return list(self.dots)

    @staticmethod
    def from_bits(bits: List[bool]) -> "BrailleCell":
        assert len(bits) == 8
        return BrailleCell(dots=list(bits))


@dataclass
class LatticePoint:
    """A point on the braille lattice: one 8-dot cell per axis."""
    cells: Dict[str, BrailleCell]
    domain: Domain

    @property
    def total_bits(self) -> int:
        return len(self.domain.axes) * 8

    def to_bitstring(self) -> List[bool]:
        bits = []
        for axis in self.domain.axes:
            bits.extend(self.cells[axis].to_bits())
        return bits

    @staticmethod
    def from_bitstring(bits: List[bool], domain: Domain) -> "LatticePoint":
        cells = {}
        idx = 0
        for axis in domain.axes:
            cells[axis] = BrailleCell.from_bits(bits[idx:idx + 8])
            idx += 8
        return LatticePoint(cells=cells, domain=domain)

    def to_unicode(self) -> str:
        return "".join(self.cells[a].to_unicode() for a in self.domain.axes)

    def __eq__(self, other):
        if not isinstance(other, LatticePoint):
            return False
        return self.to_bitstring() == other.to_bitstring()

    def __hash__(self):
        return hash(tuple(self.to_bitstring()))


def encode_to_lattice(vec: Vec, domain: Domain, prev_vec: Optional[Vec] = None) -> LatticePoint:
    """
    Project a continuous vector onto the 8-dot braille lattice.

    vec ∈ ℝ^D → 𝓛 ∈ {0,1}^(D×8)

    Args:
        vec: continuous belief vector (normalized)
        domain: semantic domain definition
        prev_vec: previous timestep vector (for momentum detection)
    """
    cells = {}
    for axis in domain.axes:
        value = vec.get(axis, 0.0)

        # Polarity (dots 1-3)
        opposite = domain.polarity_pairs.get(axis)
        opp_value = vec.get(opposite, 0.0) if opposite else 0.0
        pos_active = value > 0.3
        neg_active = opp_value > value + 0.1 if opposite else False
        tension = (pos_active and opp_value > 0.3 and
                   abs(value - opp_value) < 0.15) if opposite else False

        # Intensity (dots 4-5): quantize to 4 levels
        intensity = min(3, int(value * 4))
        dot4 = (intensity & 2) != 0
        dot5 = (intensity & 1) != 0

        # Rigidity (dot 6): strong conviction
        rigid = value > 0.7

        # Salience (dot 7): axis is contextually active (above median)
        median_val = sorted(vec.get(a, 0.0) for a in domain.axes)[domain.D // 2]
        salient = value > median_val

        # Momentum (dot 8): axis is changing (requires prev_vec)
        if prev_vec is not None:
            prev_val = prev_vec.get(axis, 0.0)
            momentum = abs(value - prev_val) > 0.05
        else:
            momentum = False

        cells[axis] = BrailleCell(dots=[
            pos_active, neg_active, tension, dot4, dot5, rigid, salient, momentum
        ])

    return LatticePoint(cells=cells, domain=domain)


def decode_from_lattice(point: LatticePoint) -> Vec:
    """
    Decode a lattice point back to an approximate continuous vector.
    This is lossy — the lattice is a compression.
    """
    vec = {}
    for axis in point.domain.axes:
        cell = point.cells[axis]
        pos_active, neg_active, tension, dot4, dot5, rigid, salient, momentum = cell.dots

        # Reconstruct intensity
        intensity = (2 if dot4 else 0) + (1 if dot5 else 0)
        value = (intensity + 0.5) / 4

        # Adjust for polarity
        if not pos_active and neg_active:
            value *= 0.3
        if tension:
            value *= 0.85

        # Rigidity boost
        if rigid:
            value = max(value, 0.75)

        # Salience boost
        if salient:
            value *= 1.1

        vec[axis] = value

    return normalize(vec, point.domain.axes)


# ═══════════════════════════════════════════════════════════════════════
# §4  HAMMING MEAN CENTROIDS (Layer 3 — Snap Dynamics)
# ═══════════════════════════════════════════════════════════════════════

def hamming_distance(a: LatticePoint, b: LatticePoint) -> int:
    """Count differing bits between two lattice points."""
    ba, bb = a.to_bitstring(), b.to_bitstring()
    return sum(1 for x, y in zip(ba, bb) if x != y)


def normalized_hamming(a: LatticePoint, b: LatticePoint) -> float:
    """Normalized Hamming distance (0 = identical, 1 = maximally different)."""
    return hamming_distance(a, b) / a.total_bits


def hamming_centroid(points: List[LatticePoint], weights: Optional[List[float]] = None) -> LatticePoint:
    """
    Hamming mean (majority-vote centroid) on the discrete lattice.

    This is the KEY innovation: unlike arithmetic means which drift
    continuously, Hamming means SNAP to valid lattice points.
    No intermediate states exist.

    For each bit position, the centroid takes the majority value.
    With prestige weights, each point's vote is weighted.
    """
    if not points:
        raise ValueError("Cannot compute centroid of empty set")

    domain = points[0].domain
    bitstrings = [p.to_bitstring() for p in points]
    n_bits = len(bitstrings[0])

    centroid_bits = []
    for i in range(n_bits):
        if weights is None:
            ones = sum(1 for bs in bitstrings if bs[i])
            centroid_bits.append(ones > len(points) / 2)
        else:
            weighted_ones = sum(w for bs, w in zip(bitstrings, weights) if bs[i])
            total_weight = sum(weights)
            centroid_bits.append(weighted_ones > total_weight / 2)

    return LatticePoint.from_bitstring(centroid_bits, domain)


def centroid_is_stable(before: LatticePoint, after: LatticePoint) -> bool:
    """
    Corollary 4 (Braille-Enforced Stability):
    The centroid is invariant under perturbations that don't flip
    any cell's majority. Returns True if no bits changed.
    """
    return hamming_distance(before, after) == 0


# ═══════════════════════════════════════════════════════════════════════
# §5  OPERATOR LAYER (Fusion, Fission, Perturbation)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OperatorResult:
    """Result of applying an operator on the lattice."""
    op_type: str
    input_points: List[LatticePoint]
    output_points: List[LatticePoint]
    flips: int  # total cell flips caused
    metadata: Dict = field(default_factory=dict)


def op_fusion(a: LatticePoint, b: LatticePoint,
              weight_a: float = 1.0, weight_b: float = 1.0) -> OperatorResult:
    """
    Fusion operator (§3.1): merge two attractors into one.
    Syncretism — e.g., Roman gods identified with Greek counterparts.

    The merged centroid is the weighted Hamming mean of the two inputs.
    """
    merged = hamming_centroid([a, b], weights=[weight_a, weight_b])
    flips_from_a = hamming_distance(a, merged)
    flips_from_b = hamming_distance(b, merged)

    return OperatorResult(
        op_type="fusion",
        input_points=[a, b],
        output_points=[merged],
        flips=flips_from_a + flips_from_b,
        metadata={
            "flips_from_a": flips_from_a,
            "flips_from_b": flips_from_b,
            "weight_a": weight_a,
            "weight_b": weight_b,
        }
    )


def op_fission(point: LatticePoint, members: List[LatticePoint],
               weights: Optional[List[float]] = None) -> OperatorResult:
    """
    Fission operator (§3.2): split one attractor into two.
    Schism — e.g., Protestant Reformation.

    Finds the two most distant members, then assigns each member
    to the nearer seed. Returns two new Hamming-mean centroids.
    """
    if len(members) < 4:
        return OperatorResult("fission", [point], [point], 0,
                              {"error": "too few members to split"})

    # Find the two most distant members
    max_dist = -1
    seed_a, seed_b = 0, 1
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            d = hamming_distance(members[i], members[j])
            if d > max_dist:
                max_dist = d
                seed_a, seed_b = i, j

    # Assign each member to nearer seed
    group_a, group_b = [], []
    weights_a, weights_b = [], []
    for k, m in enumerate(members):
        da = hamming_distance(m, members[seed_a])
        db = hamming_distance(m, members[seed_b])
        w = weights[k] if weights else 1.0
        if da <= db:
            group_a.append(m)
            weights_a.append(w)
        else:
            group_b.append(m)
            weights_b.append(w)

    if not group_a or not group_b:
        return OperatorResult("fission", [point], [point], 0,
                              {"error": "degenerate split"})

    centroid_a = hamming_centroid(group_a, weights_a)
    centroid_b = hamming_centroid(group_b, weights_b)

    return OperatorResult(
        op_type="fission",
        input_points=[point],
        output_points=[centroid_a, centroid_b],
        flips=hamming_distance(point, centroid_a) + hamming_distance(point, centroid_b),
        metadata={
            "group_sizes": (len(group_a), len(group_b)),
            "inter_group_distance": hamming_distance(centroid_a, centroid_b),
            "max_intra_distance": max_dist,
        }
    )


def op_perturbation(centroid: LatticePoint, prophet_vec: Vec,
                    pull_strength: float = 0.5) -> OperatorResult:
    """
    Perturbation operator (§3.3): prophetic revelation.
    A high-prestige agent shifts the attractor.

    Projects the prophet's continuous vector onto the lattice,
    then computes a weighted Hamming mean between the existing
    centroid and the prophet's lattice point.
    """
    prophet_lattice = encode_to_lattice(prophet_vec, centroid.domain)
    new_centroid = hamming_centroid(
        [centroid, prophet_lattice],
        weights=[1.0 - pull_strength, pull_strength]
    )

    return OperatorResult(
        op_type="perturbation",
        input_points=[centroid],
        output_points=[new_centroid],
        flips=hamming_distance(centroid, new_centroid),
        metadata={
            "pull_strength": pull_strength,
            "prophet_lattice": prophet_lattice.to_unicode(),
            "distance_prophet_to_original": hamming_distance(centroid, prophet_lattice),
        }
    )


# ═══════════════════════════════════════════════════════════════════════
# §6  CELL FLIP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

DOT_NAMES_8 = [
    "positive_polarity", "negative_polarity", "tension",
    "intensity_high", "intensity_low", "rigidity",
    "salience", "momentum"
]


@dataclass
class CellFlip:
    axis: str
    dot_index: int
    dot_name: str
    from_val: bool
    to_val: bool
    interpretation: str


def analyze_flips(before: LatticePoint, after: LatticePoint) -> List[CellFlip]:
    """Enumerate all cell flips between two lattice points."""
    flips = []
    for axis in before.domain.axes:
        cb = before.cells[axis]
        ca = after.cells[axis]
        for d in range(8):
            if cb.dots[d] != ca.dots[d]:
                direction = "activated" if ca.dots[d] else "deactivated"
                interp = f"{axis}: {DOT_NAMES_8[d]} {direction}"
                flips.append(CellFlip(
                    axis=axis, dot_index=d, dot_name=DOT_NAMES_8[d],
                    from_val=cb.dots[d], to_val=ca.dots[d],
                    interpretation=interp
                ))
    return flips


# ═══════════════════════════════════════════════════════════════════════
# §7  RECURSIVE COMPRESSION PIPELINE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CompressionResult:
    """Full pipeline result for one cluster of vectors."""
    domain: str
    n_vectors: int
    # Layer 1: continuous
    continuous_centroid: Vec
    # Layer 2: lattice projection
    lattice_points: List[LatticePoint]
    # Layer 3: Hamming centroid (snap)
    hamming_centroid_point: LatticePoint
    hamming_centroid_unicode: str
    # Comparison
    arithmetic_centroid_projected: LatticePoint
    snap_vs_arithmetic_flips: int
    reconstruction_cosine: float
    # Stability
    is_stable_under_noise: bool


def compress(vectors: List[Vec], domain: Domain,
             weights: Optional[List[float]] = None,
             noise_test: bool = True, rng: Optional[random.Random] = None) -> CompressionResult:
    """
    Full recursive compression pipeline:

      1. Compute arithmetic mean centroid (continuous, Layer 1)
      2. Project all vectors onto 8-dot braille lattice (Layer 2)
      3. Compute Hamming mean centroid (snap dynamics, Layer 3)
      4. Compare: does the snap centroid match the projected arithmetic centroid?
      5. Test stability: add noise, recompute — does the centroid change?
    """
    if rng is None:
        rng = random.Random(42)

    axes = domain.axes

    # Layer 1: continuous centroid
    continuous_centroid = arithmetic_mean(vectors, axes, weights)

    # Layer 2: project all vectors to lattice
    lattice_points = [encode_to_lattice(v, domain) for v in vectors]

    # Layer 3: Hamming mean centroid (SNAP)
    hc = hamming_centroid(lattice_points, weights)

    # Also project the arithmetic centroid for comparison
    arith_projected = encode_to_lattice(continuous_centroid, domain)

    # How many bits differ between snap centroid and projected arithmetic centroid?
    snap_vs_arith = hamming_distance(hc, arith_projected)

    # Reconstruction quality: decode the Hamming centroid back to continuous
    decoded = decode_from_lattice(hc)
    recon_cosine = cosine_sim(continuous_centroid, decoded, axes)

    # Stability test: add small noise to each vector, recompute
    is_stable = True
    if noise_test and len(vectors) >= 2:
        noisy_vecs = []
        for v in vectors:
            nv = {a: v[a] + rng.gauss(0, 0.02) for a in axes}
            noisy_vecs.append(normalize(nv, axes))
        noisy_lattice = [encode_to_lattice(nv, domain) for nv in noisy_vecs]
        noisy_hc = hamming_centroid(noisy_lattice, weights)
        is_stable = centroid_is_stable(hc, noisy_hc)

    return CompressionResult(
        domain=domain.name,
        n_vectors=len(vectors),
        continuous_centroid=continuous_centroid,
        lattice_points=lattice_points,
        hamming_centroid_point=hc,
        hamming_centroid_unicode=hc.to_unicode(),
        arithmetic_centroid_projected=arith_projected,
        snap_vs_arithmetic_flips=snap_vs_arith,
        reconstruction_cosine=recon_cosine,
        is_stable_under_noise=is_stable,
    )


# ═══════════════════════════════════════════════════════════════════════
# §8  CHANNEL INVARIANCE TEST
# ═══════════════════════════════════════════════════════════════════════

def test_channel_invariance(
    full_vec: Vec, restricted_axes: List[str], domain: Domain
) -> Dict:
    """
    Test whether a lattice point is invariant when some axes are zeroed.
    This is the Accessibility Corollary: sensory-restricted agents
    should converge to the same attractor.
    """
    # Full projection
    full_lattice = encode_to_lattice(full_vec, domain)

    # Restricted: zero out specified axes, renormalize
    restricted = {a: (0.0 if a in restricted_axes else full_vec.get(a, 0.0))
                  for a in domain.axes}
    n = math.sqrt(sum(v ** 2 for v in restricted.values())) or 1.0
    restricted = {a: restricted[a] / n for a in domain.axes}
    restricted_lattice = encode_to_lattice(restricted, domain)

    h_dist = hamming_distance(full_lattice, restricted_lattice)
    flips = analyze_flips(full_lattice, restricted_lattice)

    return {
        "full_unicode": full_lattice.to_unicode(),
        "restricted_unicode": restricted_lattice.to_unicode(),
        "hamming_distance": h_dist,
        "normalized_distance": h_dist / full_lattice.total_bits,
        "is_invariant": h_dist == 0,
        "n_flips": len(flips),
        "flips": [f.interpretation for f in flips],
    }


# ═══════════════════════════════════════════════════════════════════════
# §9  SURJECTIVITY TEST (Conjecture from paper)
# ═══════════════════════════════════════════════════════════════════════

def test_surjectivity(domain: Domain, n_samples: int = 10000,
                      rng: Optional[random.Random] = None) -> Dict:
    """
    Test the surjectivity conjecture: every possible braille configuration
    should correspond to a valid belief region.

    We sample random continuous vectors, project them, and count how many
    unique lattice points we hit. Compare to theoretical maximum (2^(D×8)).
    """
    if rng is None:
        rng = random.Random(42)

    seen = set()
    for _ in range(n_samples):
        v = rand_vec(domain.axes, rng)
        lp = encode_to_lattice(v, domain)
        seen.add(tuple(lp.to_bitstring()))

    theoretical_max = 2 ** domain.total_bits
    coverage = len(seen) / theoretical_max if theoretical_max > 0 else 0

    return {
        "domain": domain.name,
        "n_samples": n_samples,
        "unique_lattice_points": len(seen),
        "theoretical_max": theoretical_max,
        "coverage_ratio": coverage,
        "total_bits": domain.total_bits,
        "note": (f"Sampled {n_samples} random vectors, hit {len(seen)} unique "
                 f"lattice points out of 2^{domain.total_bits} = {theoretical_max} possible. "
                 f"Full surjectivity requires targeted sampling of boundary regions.")
    }


# ═══════════════════════════════════════════════════════════════════════
# §10  FULL DEMO / VALIDATION
# ═══════════════════════════════════════════════════════════════════════

# Deity priors for theology demo
DEITY_PRIORS = {
    "Zeus":    {"authority": 0.9, "transcendence": 0.7, "care": 0.3, "justice": 0.6, "wisdom": 0.4, "power": 0.95, "fertility": 0.3, "war": 0.5, "death": 0.2, "creation": 0.4, "nature": 0.6, "order": 0.7},
    "Yahweh":  {"authority": 0.95, "transcendence": 0.95, "care": 0.7, "justice": 0.9, "wisdom": 0.8, "power": 0.9, "fertility": 0.2, "war": 0.3, "death": 0.3, "creation": 0.9, "nature": 0.3, "order": 0.9},
    "Vishnu":  {"authority": 0.6, "transcendence": 0.9, "care": 0.9, "justice": 0.7, "wisdom": 0.8, "power": 0.7, "fertility": 0.5, "war": 0.2, "death": 0.3, "creation": 0.8, "nature": 0.7, "order": 0.8},
    "Odin":    {"authority": 0.7, "transcendence": 0.5, "care": 0.2, "justice": 0.4, "wisdom": 0.95, "power": 0.6, "fertility": 0.1, "war": 0.8, "death": 0.7, "creation": 0.5, "nature": 0.4, "order": 0.3},
    "Isis":    {"authority": 0.4, "transcendence": 0.6, "care": 0.95, "justice": 0.5, "wisdom": 0.7, "power": 0.5, "fertility": 0.9, "war": 0.1, "death": 0.4, "creation": 0.6, "nature": 0.8, "order": 0.5},
    "Manitou": {"authority": 0.2, "transcendence": 0.9, "care": 0.9, "justice": 0.3, "wisdom": 0.8, "power": 0.3, "fertility": 0.7, "war": 0.1, "death": 0.2, "creation": 0.6, "nature": 0.9, "order": 0.4},
}

# Normalize
for name, vec in DEITY_PRIORS.items():
    n = math.sqrt(sum(v ** 2 for v in vec.values())) or 1.0
    DEITY_PRIORS[name] = {k: v / n for k, v in vec.items()}

# Political ideology priors
POLITICAL_PRIORS = {
    "Social Democrat":  {"economic_left": 0.7, "economic_right": 0.2, "auth_state": 0.4, "lib_individual": 0.6, "progressive": 0.8, "traditional": 0.2, "globalist": 0.7, "nationalist": 0.3, "secular": 0.7, "religious": 0.3},
    "Libertarian":      {"economic_left": 0.1, "economic_right": 0.9, "auth_state": 0.1, "lib_individual": 0.95, "progressive": 0.5, "traditional": 0.4, "globalist": 0.6, "nationalist": 0.4, "secular": 0.6, "religious": 0.3},
    "Authoritarian":    {"economic_left": 0.3, "economic_right": 0.5, "auth_state": 0.95, "lib_individual": 0.1, "progressive": 0.2, "traditional": 0.8, "globalist": 0.3, "nationalist": 0.8, "secular": 0.4, "religious": 0.6},
    "Green":            {"economic_left": 0.6, "economic_right": 0.3, "auth_state": 0.3, "lib_individual": 0.7, "progressive": 0.9, "traditional": 0.1, "globalist": 0.8, "nationalist": 0.2, "secular": 0.7, "religious": 0.2},
}

for name, vec in POLITICAL_PRIORS.items():
    n = math.sqrt(sum(v ** 2 for v in vec.values())) or 1.0
    POLITICAL_PRIORS[name] = {k: v / n for k, v in vec.items()}


def run_demo():
    """Full demonstration of the recursive compression engine."""
    rng = random.Random(42)

    print("=" * 72)
    print("RECURSIVE SEMANTIC COMPRESSION ENGINE")
    print("8-Dot Braille Lattice + Hamming Centroids + Operator Layer")
    print("=" * 72)

    # ─── Demo 1: Theology ─────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("DOMAIN: THEOLOGY (12 axes × 8 dots = 96 bits)")
    print(f"{'─' * 72}")

    for name, vec in DEITY_PRIORS.items():
        lp = encode_to_lattice(vec, THEOLOGY)
        decoded = decode_from_lattice(lp)
        recon = cosine_sim(vec, decoded, THEOLOGY_AXES)
        top3 = sorted(THEOLOGY_AXES, key=lambda a: vec[a], reverse=True)[:3]
        print(f"  {name:10s}  {lp.to_unicode()}  recon={recon:.4f}  top: {', '.join(top3)}")

    # ─── Demo 2: Compression Pipeline ─────────────────────────────────
    print(f"\n{'─' * 72}")
    print("RECURSIVE COMPRESSION: Zeus cluster (6 noisy followers)")
    print(f"{'─' * 72}")

    zeus_followers = []
    for i in range(6):
        noisy = {a: DEITY_PRIORS["Zeus"][a] + rng.gauss(0, 0.1) for a in THEOLOGY_AXES}
        zeus_followers.append(normalize(noisy, THEOLOGY_AXES))

    result = compress(zeus_followers, THEOLOGY, rng=rng)
    print(f"  Continuous centroid → lattice: {result.arithmetic_centroid_projected.to_unicode()}")
    print(f"  Hamming centroid (snap):       {result.hamming_centroid_unicode}")
    print(f"  Snap vs arithmetic flips:      {result.snap_vs_arithmetic_flips}")
    print(f"  Reconstruction cosine:         {result.reconstruction_cosine:.4f}")
    print(f"  Stable under noise:            {result.is_stable_under_noise}")

    # ─── Demo 3: Operator Layer ───────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("OPERATOR LAYER")
    print(f"{'─' * 72}")

    # Fusion: Zeus + Odin → merged deity
    zeus_lp = encode_to_lattice(DEITY_PRIORS["Zeus"], THEOLOGY)
    odin_lp = encode_to_lattice(DEITY_PRIORS["Odin"], THEOLOGY)
    fusion_result = op_fusion(zeus_lp, odin_lp)
    merged = fusion_result.output_points[0]
    decoded_merged = decode_from_lattice(merged)
    top3_merged = sorted(THEOLOGY_AXES, key=lambda a: decoded_merged[a], reverse=True)[:3]
    print(f"\n  FUSION: Zeus ⊕ Odin")
    print(f"    Zeus:   {zeus_lp.to_unicode()}")
    print(f"    Odin:   {odin_lp.to_unicode()}")
    print(f"    Merged: {merged.to_unicode()}")
    print(f"    Flips:  {fusion_result.flips} ({fusion_result.metadata['flips_from_a']} from Zeus, {fusion_result.metadata['flips_from_b']} from Odin)")
    print(f"    Top 3:  {', '.join(top3_merged)}")

    # Fission: Yahweh cluster splits
    print(f"\n  FISSION: Yahweh cluster splits")
    yahweh_members = []
    for i in range(8):
        noise_level = 0.15 if i < 4 else 0.25
        noisy = {a: DEITY_PRIORS["Yahweh"][a] + rng.gauss(0, noise_level) for a in THEOLOGY_AXES}
        yahweh_members.append(normalize(noisy, THEOLOGY_AXES))

    yahweh_lattice_members = [encode_to_lattice(v, THEOLOGY) for v in yahweh_members]
    yahweh_centroid = hamming_centroid(yahweh_lattice_members)
    fission_result = op_fission(yahweh_centroid, yahweh_lattice_members)

    if len(fission_result.output_points) == 2:
        c1, c2 = fission_result.output_points
        d1 = decode_from_lattice(c1)
        d2 = decode_from_lattice(c2)
        top3_c1 = sorted(THEOLOGY_AXES, key=lambda a: d1[a], reverse=True)[:3]
        top3_c2 = sorted(THEOLOGY_AXES, key=lambda a: d2[a], reverse=True)[:3]
        print(f"    Original: {yahweh_centroid.to_unicode()}")
        print(f"    Child A:  {c1.to_unicode()}  ({fission_result.metadata['group_sizes'][0]} members)  top: {', '.join(top3_c1)}")
        print(f"    Child B:  {c2.to_unicode()}  ({fission_result.metadata['group_sizes'][1]} members)  top: {', '.join(top3_c2)}")
        print(f"    Inter-group distance: {fission_result.metadata['inter_group_distance']} bits")
    else:
        print(f"    No fission occurred: {fission_result.metadata}")

    # Perturbation: prophet shifts Vishnu
    print(f"\n  PERTURBATION: Prophet shifts Vishnu")
    prophet_belief = {a: rng.random() for a in THEOLOGY_AXES}
    prophet_belief = normalize(prophet_belief, THEOLOGY_AXES)
    vishnu_lp = encode_to_lattice(DEITY_PRIORS["Vishnu"], THEOLOGY)
    perturb_result = op_perturbation(vishnu_lp, prophet_belief, pull_strength=0.6)
    new_vishnu = perturb_result.output_points[0]
    print(f"    Before:  {vishnu_lp.to_unicode()}")
    print(f"    After:   {new_vishnu.to_unicode()}")
    print(f"    Flips:   {perturb_result.flips}")
    flips = analyze_flips(vishnu_lp, new_vishnu)
    for f in flips[:5]:
        print(f"      → {f.interpretation}")
    if len(flips) > 5:
        print(f"      ... and {len(flips) - 5} more")

    # ─── Demo 4: Cross-Domain ─────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("CROSS-DOMAIN: Political Ideology (10 axes × 8 dots = 80 bits)")
    print(f"{'─' * 72}")

    for name, vec in POLITICAL_PRIORS.items():
        lp = encode_to_lattice(vec, POLITICAL)
        decoded = decode_from_lattice(lp)
        recon = cosine_sim(vec, decoded, POLITICAL_AXES)
        top3 = sorted(POLITICAL_AXES, key=lambda a: vec[a], reverse=True)[:3]
        print(f"  {name:20s}  {lp.to_unicode()}  recon={recon:.4f}  top: {', '.join(top3)}")

    # Fusion: Social Democrat + Green
    sd_lp = encode_to_lattice(POLITICAL_PRIORS["Social Democrat"], POLITICAL)
    green_lp = encode_to_lattice(POLITICAL_PRIORS["Green"], POLITICAL)
    pol_fusion = op_fusion(sd_lp, green_lp)
    print(f"\n  FUSION: Social Democrat ⊕ Green")
    print(f"    Merged: {pol_fusion.output_points[0].to_unicode()}  ({pol_fusion.flips} total flips)")

    # ─── Demo 5: Channel Invariance ───────────────────────────────────
    print(f"\n{'─' * 72}")
    print("CHANNEL INVARIANCE (Accessibility Corollary)")
    print(f"{'─' * 72}")

    for deity_name in ["Yahweh", "Vishnu", "Odin"]:
        vec = DEITY_PRIORS[deity_name]
        result = test_channel_invariance(
            vec, ["fertility", "war", "death", "nature"], THEOLOGY
        )
        status = "INVARIANT" if result["is_invariant"] else f"{result['hamming_distance']} flips"
        print(f"  {deity_name:10s}  restrict [fertility,war,death,nature] → {status}")

    # ─── Demo 6: Surjectivity ────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("SURJECTIVITY TEST")
    print(f"{'─' * 72}")

    for domain in [PERSONALITY, POLITICAL, THEOLOGY]:
        surj = test_surjectivity(domain, n_samples=50000, rng=rng)
        print(f"  {domain.name:12s}  {surj['unique_lattice_points']:6d} unique / 2^{surj['total_bits']} possible  ({surj['n_samples']} samples)")

    # ─── Demo 7: 6-dot vs 8-dot comparison ───────────────────────────
    print(f"\n{'─' * 72}")
    print("6-DOT vs 8-DOT COMPARISON")
    print(f"{'─' * 72}")

    # Simulate 6-dot by masking dots 7-8
    for name, vec in list(DEITY_PRIORS.items())[:3]:
        lp_8dot = encode_to_lattice(vec, THEOLOGY)
        # Create 6-dot version by zeroing dots 7-8
        lp_6dot_cells = {}
        for axis in THEOLOGY_AXES:
            cell = lp_8dot.cells[axis]
            lp_6dot_cells[axis] = BrailleCell(dots=cell.dots[:6] + [False, False])
        lp_6dot = LatticePoint(cells=lp_6dot_cells, domain=THEOLOGY)

        decoded_8 = decode_from_lattice(lp_8dot)
        decoded_6 = decode_from_lattice(lp_6dot)
        recon_8 = cosine_sim(vec, decoded_8, THEOLOGY_AXES)
        recon_6 = cosine_sim(vec, decoded_6, THEOLOGY_AXES)
        info_gain = hamming_distance(lp_8dot, lp_6dot)

        print(f"  {name:10s}  6-dot recon={recon_6:.4f}  8-dot recon={recon_8:.4f}  "
              f"info gain={info_gain} bits  Δ={recon_8 - recon_6:+.4f}")

    # ─── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"  8-dot Braille lattice:     ✓ Implemented (D×8 bits per domain)")
    print(f"  Hamming mean centroids:    ✓ Snap dynamics (majority-vote)")
    print(f"  Operator layer:            ✓ Fusion, Fission, Perturbation")
    print(f"  Recursive compression:     ✓ Continuous → Lattice → Hamming centroid")
    print(f"  Cross-domain:              ✓ Theology (96-bit), Political (80-bit), Personality (40-bit)")
    print(f"  Channel invariance:        ✓ Accessibility Corollary tested")
    print(f"  6-dot vs 8-dot:            ✓ Information gain quantified")
    print()

    return True


def run_tests():
    """Validation test suite."""
    rng = random.Random(42)
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  ✓ {name}")
        else:
            failed += 1
            print(f"  ✗ {name}")

    print("=" * 72)
    print("VALIDATION TEST SUITE")
    print("=" * 72)

    # Test 1: Encoding/decoding round-trip
    print("\n─── Round-trip encoding ───")
    for name, vec in DEITY_PRIORS.items():
        lp = encode_to_lattice(vec, THEOLOGY)
        decoded = decode_from_lattice(lp)
        recon = cosine_sim(vec, decoded, THEOLOGY_AXES)
        check(f"{name} round-trip cosine > 0.85", recon > 0.85)

    # Test 2: Hamming centroid is a valid lattice point
    print("\n─── Hamming centroid validity ───")
    points = [encode_to_lattice(v, THEOLOGY) for v in DEITY_PRIORS.values()]
    hc = hamming_centroid(points)
    check("Hamming centroid has correct bit count", len(hc.to_bitstring()) == 96)
    check("Hamming centroid is a valid LatticePoint", isinstance(hc, LatticePoint))

    # Test 3: Hamming centroid snaps (no intermediate states)
    print("\n─── Snap dynamics ───")
    bits = hc.to_bitstring()
    check("All bits are boolean", all(isinstance(b, bool) for b in bits))
    check("No intermediate values", all(b in (True, False) for b in bits))

    # Test 4: Stability under small perturbation
    print("\n─── Braille-enforced stability ───")
    zeus_followers = []
    for _ in range(20):
        noisy = {a: DEITY_PRIORS["Zeus"][a] + rng.gauss(0, 0.02) for a in THEOLOGY_AXES}
        zeus_followers.append(normalize(noisy, THEOLOGY_AXES))
    lps = [encode_to_lattice(v, THEOLOGY) for v in zeus_followers]
    hc1 = hamming_centroid(lps[:10])
    hc2 = hamming_centroid(lps[10:])
    dist = hamming_distance(hc1, hc2)
    check(f"Small noise → small drift ({dist} flips)", dist < 10)

    # Test 5: Fusion is commutative
    print("\n─── Operator properties ───")
    a = encode_to_lattice(DEITY_PRIORS["Zeus"], THEOLOGY)
    b = encode_to_lattice(DEITY_PRIORS["Odin"], THEOLOGY)
    fab = op_fusion(a, b)
    fba = op_fusion(b, a)
    check("Fusion is commutative (equal weights)", fab.output_points[0] == fba.output_points[0])

    # Test 6: Fission produces two distinct children
    members = [encode_to_lattice(v, THEOLOGY) for v in DEITY_PRIORS.values()]
    centroid = hamming_centroid(members)
    fiss = op_fission(centroid, members)
    check("Fission produces 2 children", len(fiss.output_points) == 2)
    if len(fiss.output_points) == 2:
        check("Children are distinct", fiss.output_points[0] != fiss.output_points[1])

    # Test 7: Perturbation with strength=0 is identity
    vishnu = encode_to_lattice(DEITY_PRIORS["Vishnu"], THEOLOGY)
    prophet = rand_vec(THEOLOGY_AXES, rng)
    pert0 = op_perturbation(vishnu, prophet, pull_strength=0.0)
    check("Perturbation(strength=0) is identity", pert0.output_points[0] == vishnu)

    # Test 8: Cross-domain works
    print("\n─── Cross-domain ───")
    for name, vec in POLITICAL_PRIORS.items():
        lp = encode_to_lattice(vec, POLITICAL)
        check(f"Political '{name}' encodes to {POLITICAL.total_bits} bits",
              len(lp.to_bitstring()) == POLITICAL.total_bits)

    # Test 9: Unicode rendering
    print("\n─── Unicode rendering ───")
    for name, vec in list(DEITY_PRIORS.items())[:3]:
        lp = encode_to_lattice(vec, THEOLOGY)
        unicode_str = lp.to_unicode()
        check(f"{name} renders to {len(unicode_str)} chars", len(unicode_str) == 12)

    print(f"\n{'=' * 72}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 72}")
    return failed == 0


# ═══════════════════════════════════════════════════════════════════════
# §11  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        run_demo()
