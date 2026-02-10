#!/usr/bin/env python3
"""
4-LLM Lattice Consensus Signatures
====================================

The RIGHT way to compute braille signatures from multi-model scores:

  1. Each LLM scores 126 passages → continuous vectors
  2. Each LLM's scores are projected INDEPENDENTLY to 8-dot braille lattice
  3. Per-passage: Hamming mean across 4 models' lattice points (bit-level vote)
  4. Per-tradition: Hamming mean across all passages in that tradition
  5. Result: one defensible 96-bit braille signature per tradition

This is superior to "average continuous scores then project" because:
  - Snap dynamics cancel out individual model noise at the BIT level
  - No quantization boundary artifacts from averaging before projection
  - Each model's discrete judgment is preserved and voted on

Usage:
  python sim/lattice_consensus.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────

AXES = [
    "authority", "transcendence", "care", "justice", "wisdom", "power",
    "fertility", "war", "death", "creation", "nature", "order"
]

POLARITY_PAIRS = {
    "authority": "care", "care": "authority",
    "transcendence": "nature", "nature": "transcendence",
    "justice": "fertility", "fertility": "justice",
    "wisdom": "war", "war": "wisdom",
    "power": "death", "death": "power",
    "creation": "order", "order": "creation",
}

DOT_NAMES = [
    "positive_polarity", "negative_polarity", "tension",
    "intensity_high", "intensity_low", "rigidity",
    "salience", "momentum"
]


# ─── Lattice functions ───────────────────────────────────────────────

def normalize(v):
    n = math.sqrt(sum(v.get(a, 0) ** 2 for a in AXES)) or 1.0
    return {a: v.get(a, 0) / n for a in AXES}


def encode_to_lattice(vec):
    """Encode continuous vec → list of 12 × 8-bit cells."""
    cells = []
    sorted_vals = sorted(vec.get(a, 0.0) for a in AXES)
    median_val = sorted_vals[len(AXES) // 2]
    for axis in AXES:
        value = vec.get(axis, 0.0)
        opposite = POLARITY_PAIRS.get(axis)
        opp_value = vec.get(opposite, 0.0) if opposite else 0.0
        pos_active = value > 0.3
        neg_active = opp_value > value + 0.1 if opposite else False
        tension = (pos_active and opp_value > 0.3 and
                   abs(value - opp_value) < 0.15) if opposite else False
        intensity = min(3, int(value * 4))
        dot4 = (intensity & 2) != 0
        dot5 = (intensity & 1) != 0
        rigid = value > 0.7
        salient = value > median_val
        momentum = False
        cells.append([pos_active, neg_active, tension, dot4, dot5,
                      rigid, salient, momentum])
    return cells


def cells_to_bits(cells):
    return [b for c in cells for b in c]


def bits_to_cells(bits):
    return [bits[i:i + 8] for i in range(0, len(bits), 8)]


def hamming_distance(ca, cb):
    ba, bb = cells_to_bits(ca), cells_to_bits(cb)
    return sum(1 for x, y in zip(ba, bb) if x != y)


def hamming_mean(all_cells, weights=None):
    """Majority-vote centroid across multiple lattice points."""
    n = len(all_cells)
    if n == 0:
        return [[False] * 8 for _ in range(12)]
    all_bits = [cells_to_bits(c) for c in all_cells]
    n_bits = len(all_bits[0])
    result = []
    for i in range(n_bits):
        if weights is None:
            ones = sum(1 for bs in all_bits if bs[i])
            result.append(ones > n / 2)
        else:
            w_ones = sum(w for bs, w in zip(all_bits, weights) if bs[i])
            w_total = sum(weights)
            result.append(w_ones > w_total / 2)
    return bits_to_cells(result)


def cells_to_unicode(cells):
    offsets = [1, 2, 4, 8, 16, 32, 64, 128]
    chars = []
    for cell in cells:
        code = 0x2800
        for i, d in enumerate(cell):
            if d:
                code += offsets[i]
        chars.append(chr(code))
    return "".join(chars)


def decode_from_lattice(cells):
    """Decode lattice cells back to approximate continuous vector."""
    vec = {}
    for i, axis in enumerate(AXES):
        cell = cells[i]
        pos_active, neg_active, tension, dot4, dot5, rigid, salient, momentum = cell
        intensity = (2 if dot4 else 0) + (1 if dot5 else 0)
        value = (intensity + 0.5) / 4
        if not pos_active and neg_active:
            value *= 0.3
        if tension:
            value *= 0.85
        if rigid:
            value = max(value, 0.75)
        if salient:
            value *= 1.1
        vec[axis] = value
    return normalize(vec)


def cosine_sim(a, b):
    na = math.sqrt(sum(a.get(k, 0) ** 2 for k in AXES)) or 1.0
    nb = math.sqrt(sum(b.get(k, 0) ** 2 for k in AXES)) or 1.0
    return sum(a.get(k, 0) * b.get(k, 0) for k in AXES) / (na * nb)


# ─── Load all 4 scorer files ─────────────────────────────────────────

def load_scorers():
    base = Path("mlx-pipeline")
    scorers = {}

    # Claude (baseline)
    claude_path = base / "expanded_embeddings_results.json"
    if claude_path.exists():
        with open(claude_path) as f:
            scorers["Claude"] = json.load(f)["embeddings"]
    else:
        print(f"WARNING: {claude_path} not found")

    # GPT-4o
    gpt_path = base / "scores_gpt4o.json"
    if gpt_path.exists():
        with open(gpt_path) as f:
            scorers["GPT-4o"] = json.load(f)["embeddings"]

    # Gemini Flash
    gemini_path = base / "scores_gemini_flash.json"
    if gemini_path.exists():
        with open(gemini_path) as f:
            scorers["Gemini"] = json.load(f)["embeddings"]

    # Llama 70B
    llama_path = base / "scores_llama70b.json"
    if llama_path.exists():
        with open(llama_path) as f:
            scorers["Llama-70B"] = json.load(f)["embeddings"]

    return scorers


# ─── Main pipeline ───────────────────────────────────────────────────

def main():
    scorers = load_scorers()
    n_scorers = len(scorers)
    scorer_names = sorted(scorers.keys())

    print("=" * 80)
    print("4-LLM LATTICE CONSENSUS SIGNATURES")
    print(f"Models: {', '.join(scorer_names)}")
    print("=" * 80)

    n_passages = len(next(iter(scorers.values())))
    print(f"\n{n_passages} passages × {n_scorers} models = {n_passages * n_scorers} lattice projections")

    # ─── Step 1: Project each model's scores independently ────────
    print(f"\nStep 1: Project each model's scores to 8-dot braille lattice...")

    # per_model_lattice[model][passage_idx] = cells
    per_model_lattice = {}
    for model_name, embeddings in scorers.items():
        lattice_points = []
        for e in embeddings:
            vec = normalize(e.get("normalized", e.get("raw_scores", {})))
            cells = encode_to_lattice(vec)
            lattice_points.append(cells)
        per_model_lattice[model_name] = lattice_points
        print(f"  {model_name}: {len(lattice_points)} passages projected")

    # ─── Step 2: Per-passage Hamming mean across 4 models ─────────
    print(f"\nStep 2: Hamming mean across {n_scorers} models per passage (bit-level vote)...")

    passage_consensus = []  # one lattice point per passage
    per_passage_agreement = []  # how many bits all 4 agree on

    for i in range(n_passages):
        model_cells = [per_model_lattice[m][i] for m in scorer_names]
        consensus_cells = hamming_mean(model_cells)
        passage_consensus.append(consensus_cells)

        # Measure agreement: for each bit, count how many models agree with consensus
        consensus_bits = cells_to_bits(consensus_cells)
        all_model_bits = [cells_to_bits(mc) for mc in model_cells]
        unanimous = sum(1 for j in range(96)
                        if all(mb[j] == consensus_bits[j] for mb in all_model_bits))
        per_passage_agreement.append(unanimous)

    mean_unanimous = sum(per_passage_agreement) / len(per_passage_agreement)
    print(f"  Mean unanimous bits per passage: {mean_unanimous:.1f} / 96 ({mean_unanimous/96*100:.1f}%)")
    print(f"  Min unanimous: {min(per_passage_agreement)} / 96")
    print(f"  Max unanimous: {max(per_passage_agreement)} / 96")

    # ─── Step 3: Per-tradition Hamming centroid ───────────────────
    print(f"\nStep 3: Per-tradition Hamming centroid from passage-level consensus...")

    # Get tradition info from first scorer
    first_scorer = next(iter(scorers.values()))
    tradition_passages = defaultdict(list)
    tradition_clusters = {}
    for i, e in enumerate(first_scorer):
        tradition_passages[e["tradition"]].append(i)
        tradition_clusters[e["tradition"]] = e["expected_cluster"]

    tradition_sigs = {}
    for tradition in sorted(tradition_passages.keys()):
        indices = tradition_passages[tradition]
        member_cells = [passage_consensus[i] for i in indices]
        centroid = hamming_mean(member_cells)
        tradition_sigs[tradition] = {
            "cells": centroid,
            "unicode": cells_to_unicode(centroid),
            "n_passages": len(indices),
            "cluster": tradition_clusters[tradition],
        }

    # ─── Step 4: Compare with continuous-consensus approach ───────
    print(f"\nStep 4: Compare lattice-consensus vs continuous-consensus...")

    # Load continuous consensus
    with open("mlx-pipeline/multi_scorer_consensus.json") as f:
        continuous_consensus = json.load(f)

    # Compute continuous-consensus tradition centroids
    cont_by_trad = defaultdict(list)
    for e in continuous_consensus["embeddings"]:
        vec = normalize(e.get("normalized", e.get("raw_scores", {})))
        cont_by_trad[e["tradition"]].append(vec)

    cont_tradition_sigs = {}
    for tradition, vecs in cont_by_trad.items():
        # Average then project
        avg = {a: sum(v[a] for v in vecs) / len(vecs) for a in AXES}
        avg = normalize(avg)
        cells = encode_to_lattice(avg)
        cont_tradition_sigs[tradition] = {
            "cells": cells,
            "unicode": cells_to_unicode(cells),
        }

    # ─── Step 5: Per-model agreement with lattice consensus ───────
    print(f"\nStep 5: Per-model agreement with lattice consensus...")

    model_agreement = {}
    for model_name in scorer_names:
        trad_dists = []
        for tradition in sorted(tradition_sigs.keys()):
            indices = tradition_passages[tradition]
            model_member_cells = [per_model_lattice[model_name][i] for i in indices]
            model_centroid = hamming_mean(model_member_cells)
            consensus_centroid = tradition_sigs[tradition]["cells"]
            dist = hamming_distance(model_centroid, consensus_centroid)
            trad_dists.append(dist)
        model_agreement[model_name] = {
            "mean_dist": sum(trad_dists) / len(trad_dists),
            "max_dist": max(trad_dists),
            "min_dist": min(trad_dists),
            "exact_matches": sum(1 for d in trad_dists if d == 0),
        }
        print(f"  {model_name:<12} mean={model_agreement[model_name]['mean_dist']:.1f} bits  "
              f"exact={model_agreement[model_name]['exact_matches']}/{len(trad_dists)}  "
              f"max={model_agreement[model_name]['max_dist']}")

    # ─── Output: Full signature table ─────────────────────────────
    print(f"\n{'=' * 80}")
    print("FINAL 4-LLM LATTICE CONSENSUS SIGNATURES")
    print(f"{'=' * 80}")
    print(f"\n{'Tradition':<25} {'Cluster':<22} {'Lattice Consensus':>14}  "
          f"{'Cont.Cons':>14}  {'Δ':>3}  {'Top 3 Axes'}")
    print("-" * 115)

    all_deltas = []
    rows_by_cluster = defaultdict(list)

    for tradition in sorted(tradition_sigs.keys()):
        lc = tradition_sigs[tradition]
        cc = cont_tradition_sigs.get(tradition)
        delta = hamming_distance(lc["cells"], cc["cells"]) if cc else -1
        all_deltas.append(delta)

        decoded = decode_from_lattice(lc["cells"])
        top3 = sorted(AXES, key=lambda a: decoded[a], reverse=True)[:3]

        rows_by_cluster[lc["cluster"]].append({
            "tradition": tradition,
            "cluster": lc["cluster"],
            "lattice_unicode": lc["unicode"],
            "cont_unicode": cc["unicode"] if cc else "?",
            "delta": delta,
            "top3": ", ".join(top3),
            "n_passages": lc["n_passages"],
        })

    for cluster in sorted(rows_by_cluster.keys()):
        print(f"\n  [{cluster}]")
        for row in sorted(rows_by_cluster[cluster], key=lambda r: r["tradition"]):
            print(f"  {row['tradition']:<23} {row['cluster']:<22} "
                  f"{row['lattice_unicode']:>14}  {row['cont_unicode']:>14}  "
                  f"{row['delta']:>3}  {row['top3']}")

    # ─── Per-bit agreement analysis ───────────────────────────────
    print(f"\n{'=' * 80}")
    print("PER-DOT AGREEMENT ANALYSIS")
    print(f"{'=' * 80}")
    print(f"\nFor each dot position, what fraction of traditions have all 4 models agree?")

    dot_agreement = {d: 0 for d in range(8)}
    dot_total = {d: 0 for d in range(8)}

    for tradition in sorted(tradition_sigs.keys()):
        indices = tradition_passages[tradition]
        for passage_idx in indices:
            model_cells_list = [per_model_lattice[m][passage_idx] for m in scorer_names]
            for axis_idx in range(12):
                for dot_idx in range(8):
                    vals = [mc[axis_idx][dot_idx] for mc in model_cells_list]
                    dot_total[dot_idx] += 1
                    if all(v == vals[0] for v in vals):
                        dot_agreement[dot_idx] += 1

    print(f"\n  {'Dot':<25} {'Agreement':>10} {'Rate':>8}")
    print(f"  {'-' * 45}")
    for d in range(8):
        rate = dot_agreement[d] / dot_total[d] if dot_total[d] > 0 else 0
        bar = "█" * int(rate * 30)
        print(f"  {DOT_NAMES[d]:<25} {dot_agreement[d]:>6}/{dot_total[d]:<6} {rate:>7.1%}  {bar}")

    # ─── Summary stats ────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Models:                    {', '.join(scorer_names)}")
    print(f"  Passages:                  {n_passages}")
    print(f"  Traditions:                {len(tradition_sigs)}")
    print(f"  Mean unanimous bits/passage: {mean_unanimous:.1f} / 96 ({mean_unanimous/96*100:.1f}%)")
    print(f"  Mean Δ (lattice vs cont):  {sum(all_deltas)/len(all_deltas):.1f} bits")
    print(f"  Max Δ:                     {max(all_deltas)} bits")
    print(f"  Min Δ:                     {min(all_deltas)} bits")
    print(f"  Traditions with Δ=0:       {sum(1 for d in all_deltas if d == 0)}/{len(all_deltas)}")

    for model_name in scorer_names:
        ma = model_agreement[model_name]
        print(f"  {model_name} → consensus:    mean={ma['mean_dist']:.1f} bits, "
              f"exact={ma['exact_matches']}/{len(tradition_sigs)}")

    # ─── Save results ─────────────────────────────────────────────
    output_dir = Path("sim/runs/lattice_consensus")
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "description": "4-LLM lattice consensus signatures (Hamming mean at bit level)",
        "models": scorer_names,
        "n_passages": n_passages,
        "n_traditions": len(tradition_sigs),
        "mean_unanimous_bits": round(mean_unanimous, 1),
        "model_agreement": model_agreement,
        "per_dot_agreement": {
            DOT_NAMES[d]: round(dot_agreement[d] / dot_total[d], 4) if dot_total[d] > 0 else 0
            for d in range(8)
        },
        "traditions": {
            tradition: {
                "unicode": data["unicode"],
                "cluster": data["cluster"],
                "n_passages": data["n_passages"],
                "bits": cells_to_bits(data["cells"]),
                "top3_axes": sorted(AXES, key=lambda a: decode_from_lattice(data["cells"]).get(a, 0), reverse=True)[:3],
                "delta_vs_continuous": hamming_distance(
                    data["cells"],
                    cont_tradition_sigs[tradition]["cells"]
                ) if tradition in cont_tradition_sigs else -1,
            }
            for tradition, data in tradition_sigs.items()
        },
    }

    output_path = output_dir / "lattice_consensus.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    print()


if __name__ == "__main__":
    main()
