#!/usr/bin/env python3
"""
Corpus-to-Lattice Pipeline on Modal
====================================

Takes the 126 consensus-scored passages (4-model mean), projects them
onto the 8-dot braille lattice, computes Hamming-mean centroids per
tradition, and compares to the arithmetic centroids.

Key questions answered:
  1. Do traditions that cluster in continuous space also cluster on the lattice?
  2. How many bits separate traditions within vs across expected clusters?
  3. Does the fission threshold (σ²_max) still discriminate on the lattice?
  4. What is the per-tradition braille signature?

Usage:
  modal run sim/corpus_lattice_modal.py
"""

import modal
import json
import math
import os

app = modal.App("corpus-lattice-pipeline")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("matplotlib", "numpy")
)

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


# ─── Per-tradition analysis (one Modal container per expected_cluster) ─

@app.function(image=image, timeout=1800, cpu=1)
def analyze_cluster(cluster_name: str, passages: list):
    """
    Analyze one expected_cluster group:
      - Project all passages to 8-dot lattice
      - Compute Hamming centroid per tradition within cluster
      - Compute cluster-level Hamming centroid
      - Measure intra/inter-tradition Hamming distances
      - Compare to arithmetic centroid
    """
    import numpy as np
    from collections import defaultdict

    # ─── Inline utilities ─────────────────────────────────────────
    def _norm(a):
        return math.sqrt(sum(a.get(k, 0) ** 2 for k in AXES))

    def _normalize(a):
        n = _norm(a) or 1.0
        return {k: a.get(k, 0) / n for k in AXES}

    def _cosine(a, b):
        na, nb = _norm(a), _norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return sum(a.get(k, 0) * b.get(k, 0) for k in AXES) / (na * nb)

    # ─── 8-dot Braille lattice ────────────────────────────────────
    def encode_to_lattice(vec):
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

    def decode_from_lattice(cells):
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
        return _normalize(vec)

    def cells_to_bits(cells):
        bits = []
        for cell in cells:
            bits.extend(cell)
        return bits

    def bits_to_cells(bits):
        cells = []
        for i in range(0, len(bits), 8):
            cells.append(list(bits[i:i+8]))
        return cells

    def hamming_distance(ca, cb):
        ba, bb = cells_to_bits(ca), cells_to_bits(cb)
        return sum(1 for x, y in zip(ba, bb) if x != y)

    def hamming_mean(all_cells, weights=None):
        n = len(all_cells)
        if n == 0:
            return [[False]*8 for _ in range(12)]
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
        chars = []
        offsets = [1, 2, 4, 8, 16, 32, 64, 128]
        for cell in cells:
            code = 0x2800
            for i, d in enumerate(cell):
                if d:
                    code += offsets[i]
            chars.append(chr(code))
        return "".join(chars)

    # ─── Group passages by tradition ──────────────────────────────
    by_tradition = defaultdict(list)
    for p in passages:
        by_tradition[p["tradition"]].append(p)

    # ─── Project all passages ─────────────────────────────────────
    all_lattice_points = []
    tradition_results = {}

    for tradition, trad_passages in by_tradition.items():
        trad_cells = []
        trad_vecs = []
        for p in trad_passages:
            vec = p.get("normalized", p.get("raw_scores", {}))
            vec = _normalize(vec)
            cells = encode_to_lattice(vec)
            trad_cells.append(cells)
            trad_vecs.append(vec)
            all_lattice_points.append(cells)

        # Hamming centroid for this tradition
        ham_centroid = hamming_mean(trad_cells)
        ham_unicode = cells_to_unicode(ham_centroid)
        ham_decoded = decode_from_lattice(ham_centroid)

        # Arithmetic centroid for comparison
        arith = {k: 0.0 for k in AXES}
        for v in trad_vecs:
            for k in AXES:
                arith[k] += v.get(k, 0)
        arith = _normalize(arith)
        arith_cells = encode_to_lattice(arith)
        arith_unicode = cells_to_unicode(arith_cells)

        # Hamming distance between Hamming centroid and arithmetic centroid
        snap_distance = hamming_distance(ham_centroid, arith_cells)

        # Intra-tradition variance (Hamming)
        intra_dists = []
        for c in trad_cells:
            intra_dists.append(hamming_distance(c, ham_centroid))
        mean_intra = np.mean(intra_dists) if intra_dists else 0
        var_intra = np.var(intra_dists) if intra_dists else 0

        # Top 3 axes
        top3 = sorted(AXES, key=lambda a: ham_decoded.get(a, 0), reverse=True)[:3]

        # Cosine between arithmetic and Hamming-decoded centroids
        recon_cosine = _cosine(arith, ham_decoded)

        tradition_results[tradition] = {
            "n_passages": len(trad_passages),
            "hamming_centroid_unicode": ham_unicode,
            "arithmetic_centroid_unicode": arith_unicode,
            "snap_distance": snap_distance,
            "mean_intra_hamming": float(mean_intra),
            "var_intra_hamming": float(var_intra),
            "top3_axes": top3,
            "recon_cosine": float(recon_cosine),
            "hamming_centroid_decoded": ham_decoded,
            "arithmetic_centroid": arith,
        }

    # ─── Cluster-level centroid ───────────────────────────────────
    cluster_ham = hamming_mean(all_lattice_points)
    cluster_unicode = cells_to_unicode(cluster_ham)

    # Inter-tradition distances within this cluster
    traditions = list(tradition_results.keys())
    inter_dists = []
    for i in range(len(traditions)):
        for j in range(i + 1, len(traditions)):
            ci = encode_to_lattice(tradition_results[traditions[i]]["hamming_centroid_decoded"])
            cj = encode_to_lattice(tradition_results[traditions[j]]["hamming_centroid_decoded"])
            inter_dists.append({
                "pair": f"{traditions[i]} × {traditions[j]}",
                "hamming": hamming_distance(ci, cj),
                "cosine": float(_cosine(
                    tradition_results[traditions[i]]["hamming_centroid_decoded"],
                    tradition_results[traditions[j]]["hamming_centroid_decoded"]
                )),
            })

    # Clean up non-serializable data
    for t in tradition_results:
        tradition_results[t]["hamming_centroid_decoded"] = {
            k: round(v, 4) for k, v in tradition_results[t]["hamming_centroid_decoded"].items()
        }
        tradition_results[t]["arithmetic_centroid"] = {
            k: round(v, 4) for k, v in tradition_results[t]["arithmetic_centroid"].items()
        }

    return {
        "cluster": cluster_name,
        "n_traditions": len(traditions),
        "n_passages": len(passages),
        "cluster_centroid_unicode": cluster_unicode,
        "traditions": tradition_results,
        "inter_tradition_distances": inter_dists,
    }


# ─── Plot generation ─────────────────────────────────────────────────

@app.function(image=image, timeout=1800)
def generate_lattice_plots(all_results: list):
    """Generate visualization of lattice analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import io

    # Collect all traditions
    all_traditions = []
    for r in all_results:
        for trad_name, trad_data in r["traditions"].items():
            all_traditions.append({
                "tradition": trad_name,
                "cluster": r["cluster"],
                "snap_distance": trad_data["snap_distance"],
                "mean_intra_hamming": trad_data["mean_intra_hamming"],
                "var_intra_hamming": trad_data["var_intra_hamming"],
                "recon_cosine": trad_data["recon_cosine"],
                "unicode": trad_data["hamming_centroid_unicode"],
                "n_passages": trad_data["n_passages"],
            })

    # Sort by cluster then tradition
    all_traditions.sort(key=lambda x: (x["cluster"], x["tradition"]))

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Plot 1: Snap distance (Hamming vs Arithmetic centroid)
    ax = axes[0][0]
    names = [t["tradition"][:15] for t in all_traditions]
    snaps = [t["snap_distance"] for t in all_traditions]
    colors = []
    cluster_names = sorted(set(t["cluster"] for t in all_traditions))
    cmap = plt.cm.Set2(np.linspace(0, 1, len(cluster_names)))
    cluster_color = {c: cmap[i] for i, c in enumerate(cluster_names)}
    for t in all_traditions:
        colors.append(cluster_color[t["cluster"]])
    ax.barh(range(len(names)), snaps, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Snap distance (bits)", fontsize=11)
    ax.set_title("Hamming vs Arithmetic Centroid Distance", fontsize=12, fontweight='bold')
    ax.invert_yaxis()

    # Plot 2: Intra-tradition variance (Hamming)
    ax = axes[0][1]
    vars_h = [t["var_intra_hamming"] for t in all_traditions]
    ax.barh(range(len(names)), vars_h, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Intra-tradition Hamming variance", fontsize=11)
    ax.set_title("Lattice Variance (Fission Discriminator)", fontsize=12, fontweight='bold')
    ax.invert_yaxis()

    # Plot 3: Reconstruction cosine
    ax = axes[1][0]
    recons = [t["recon_cosine"] for t in all_traditions]
    ax.barh(range(len(names)), recons, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Cosine similarity", fontsize=11)
    ax.set_title("Lattice Round-Trip Reconstruction Quality", fontsize=12, fontweight='bold')
    ax.set_xlim(0.8, 1.0)
    ax.invert_yaxis()

    # Plot 4: Inter-cluster distance heatmap
    ax = axes[1][1]
    # Collect cluster-level centroids
    cluster_centroids = {}
    for r in all_results:
        cluster_centroids[r["cluster"]] = r["cluster_centroid_unicode"]

    # Use inter-tradition distances to build a summary
    all_inter = []
    for r in all_results:
        for d in r["inter_tradition_distances"]:
            all_inter.append(d)

    if all_inter:
        ax.text(0.5, 0.5, f"{len(all_inter)} inter-tradition\ndistance pairs computed",
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        mean_inter = np.mean([d["hamming"] for d in all_inter])
        ax.text(0.5, 0.3, f"Mean inter-tradition: {mean_inter:.1f} bits",
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.set_title("Inter-Tradition Distances", fontsize=12, fontweight='bold')

    # Legend for clusters
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cluster_color[c], label=c) for c in cluster_names]
    fig.legend(handles=legend_elements, loc='lower center', ncol=min(6, len(cluster_names)),
               fontsize=8, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("126 Passages → 8-Dot Braille Lattice Analysis",
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf.read()


# ─── Orchestrator ────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    # Load consensus data
    consensus_path = "mlx-pipeline/multi_scorer_consensus.json"
    if not os.path.exists(consensus_path):
        print(f"ERROR: {consensus_path} not found. Run multi_scorer.py first.")
        return

    with open(consensus_path) as f:
        consensus = json.load(f)

    passages = consensus["embeddings"]
    print(f"Loaded {len(passages)} passages")

    # Group by expected_cluster
    from collections import defaultdict
    by_cluster = defaultdict(list)
    for p in passages:
        by_cluster[p["expected_cluster"]].append(p)

    print(f"Found {len(by_cluster)} expected clusters:")
    for cluster, plist in sorted(by_cluster.items()):
        traditions = set(p["tradition"] for p in plist)
        print(f"  {cluster}: {len(plist)} passages, {len(traditions)} traditions")

    # Launch one container per cluster (parallel)
    print(f"\nLaunching {len(by_cluster)} parallel analysis tasks on Modal...")
    futures = []
    for cluster_name, cluster_passages in by_cluster.items():
        futures.append(
            analyze_cluster.spawn(cluster_name, cluster_passages)
        )

    # Collect results
    all_results = []
    for f in futures:
        result = f.get()
        all_results.append(result)
        print(f"  Cluster '{result['cluster']}' complete: "
              f"{result['n_traditions']} traditions, {result['n_passages']} passages")

    # Generate plots
    print("\nGenerating lattice analysis plots...")
    plot_bytes = generate_lattice_plots.remote(all_results)

    # Save everything
    output_dir = "sim/runs/corpus_lattice"
    os.makedirs(output_dir, exist_ok=True)

    # Save plot
    plot_path = os.path.join(output_dir, "corpus_lattice_analysis.png")
    with open(plot_path, "wb") as f:
        f.write(plot_bytes)
    print(f"Plot saved to {plot_path}")

    # Save raw data
    data_path = os.path.join(output_dir, "corpus_lattice_data.json")
    with open(data_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Data saved to {data_path}")

    # ─── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("CORPUS → 8-DOT BRAILLE LATTICE ANALYSIS")
    print(f"{'=' * 72}")

    print(f"\n{'Tradition':<25} {'Braille':>14} {'Snap':>5} {'Var':>6} {'Recon':>6} {'Top 3 Axes'}")
    print("-" * 85)

    for r in sorted(all_results, key=lambda x: x["cluster"]):
        print(f"\n  [{r['cluster']}]")
        for trad_name, trad_data in sorted(r["traditions"].items()):
            top3 = ", ".join(trad_data["top3_axes"])
            print(f"  {trad_name:<23} {trad_data['hamming_centroid_unicode']:>14} "
                  f"{trad_data['snap_distance']:>5} "
                  f"{trad_data['var_intra_hamming']:>6.2f} "
                  f"{trad_data['recon_cosine']:>6.4f} "
                  f"{top3}")

    # Aggregate stats
    all_snaps = []
    all_recons = []
    all_vars = []
    for r in all_results:
        for trad_data in r["traditions"].values():
            all_snaps.append(trad_data["snap_distance"])
            all_recons.append(trad_data["recon_cosine"])
            all_vars.append(trad_data["var_intra_hamming"])

    import statistics
    print(f"\n{'=' * 72}")
    print("AGGREGATE STATISTICS")
    print(f"{'=' * 72}")
    print(f"  Mean snap distance:        {statistics.mean(all_snaps):.2f} bits")
    print(f"  Mean reconstruction cos:   {statistics.mean(all_recons):.4f}")
    print(f"  Mean intra-trad variance:  {statistics.mean(all_vars):.2f}")
    print(f"  Max intra-trad variance:   {max(all_vars):.2f}")
    print(f"  Min intra-trad variance:   {min(all_vars):.2f}")

    # Fission discriminator test
    print(f"\n{'=' * 72}")
    print("FISSION DISCRIMINATOR (lattice variance)")
    print(f"{'=' * 72}")
    known_schism = {"Hinduism", "Buddhism", "Christianity", "Islam"}
    known_stable = {"Jainism", "Sikhism", "Zoroastrianism", "Cao Dai"}
    schism_vars = []
    stable_vars = []
    for r in all_results:
        for trad_name, trad_data in r["traditions"].items():
            if trad_name in known_schism:
                schism_vars.append((trad_name, trad_data["var_intra_hamming"]))
            elif trad_name in known_stable:
                stable_vars.append((trad_name, trad_data["var_intra_hamming"]))

    if schism_vars:
        print("  Known schism traditions:")
        for name, v in sorted(schism_vars, key=lambda x: -x[1]):
            print(f"    {name:<20} var={v:.2f}")
    if stable_vars:
        print("  Known stable traditions:")
        for name, v in sorted(stable_vars, key=lambda x: -x[1]):
            print(f"    {name:<20} var={v:.2f}")
    if schism_vars and stable_vars:
        mean_schism = statistics.mean([v for _, v in schism_vars])
        mean_stable = statistics.mean([v for _, v in stable_vars])
        print(f"\n  Mean schism variance:  {mean_schism:.2f}")
        print(f"  Mean stable variance:  {mean_stable:.2f}")
        if mean_schism > mean_stable:
            print("  ✓ Fission discriminator works on the lattice!")
        else:
            print("  ~ Fission discriminator inconclusive on lattice")

    print()
