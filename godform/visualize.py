#!/usr/bin/env python3
"""
Godform Trajectory Visualization
=================================
Plots 96-bit braille lattice trajectories across braiding rounds using:
  1. PCA projection of per-model lattice codes → 2D scatter per round
  2. T-SNE projection showing convergence clustering
  3. Bit-flip heatmap across rounds (96 bits × N rounds)
  4. Axis evolution radar chart
  5. Entropy + bit-flip convergence timeline

Usage:
  python godform/visualize.py godform/runs/godform_modal_20260211_122301.json
  python godform/visualize.py godform/runs/*.json  # compare multiple runs
"""

from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors

# ─── Constants ────────────────────────────────────────────────────────

DEFAULT_AXES = [
    "authority", "transcendence", "care", "justice", "wisdom", "power",
    "fertility", "war", "death", "creation", "nature", "order"
]


def get_axes(run: dict) -> list:
    """Get axes from run data, falling back to theology defaults."""
    return run.get("axes", DEFAULT_AXES)


def get_n_bits(run: dict) -> int:
    return len(get_axes(run)) * 8

# Braille Unicode → bit array conversion
BRAILLE_BASE = 0x2800

def braille_to_bits(braille_str: str) -> np.ndarray:
    """Convert a 12-char braille Unicode string to a 96-bit array."""
    bits = []
    for ch in braille_str:
        code = ord(ch) - BRAILLE_BASE
        for bit_idx in range(8):
            bits.append((code >> bit_idx) & 1)
    return np.array(bits, dtype=np.float32)


def load_run(path: str) -> dict:
    """Load a Godform run JSON file."""
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# PLOT 1: PCA/T-SNE of per-model lattice codes across rounds
# ═══════════════════════════════════════════════════════════════════════

def extract_model_lattices(run: dict) -> dict:
    """
    Extract per-model lattice bit vectors for each round and prompt.
    Returns: {round_idx: {model_name: [96-bit arrays]}}
    Also returns meta-godform lattices per round.
    """
    rounds_data = {}
    meta_lattices = []

    for rd in run["trajectory"]:
        round_idx = rd["round"]
        model_vecs = {}

        for pg in rd.get("prompt_godforms", []):
            # Per-prompt godform (consensus)
            for ma in pg.get("model_analysis", []):
                model = ma["model"]
                lattice_unicode = ma["lattice_unicode"]
                bits = braille_to_bits(lattice_unicode)
                if model not in model_vecs:
                    model_vecs[model] = []
                model_vecs[model].append(bits)

        rounds_data[round_idx] = model_vecs

        # Meta-godform
        meta_unicode = rd.get("meta_godform_unicode", rd.get("godform_sign", ""))
        if meta_unicode:
            meta_lattices.append(braille_to_bits(meta_unicode))

    return rounds_data, meta_lattices


def plot_pca_trajectory(run: dict, ax: plt.Axes, use_tsne: bool = False):
    """Plot PCA or T-SNE of all model lattice codes, colored by round."""
    rounds_data, meta_lattices = extract_model_lattices(run)

    # Collect all vectors with metadata
    all_vecs = []
    labels = []  # (round, model)

    for round_idx in sorted(rounds_data.keys()):
        for model, vecs in rounds_data[round_idx].items():
            for v in vecs:
                all_vecs.append(v)
                labels.append((round_idx, model))

    if not all_vecs:
        ax.text(0.5, 0.5, "No model lattice data", ha="center", va="center",
                transform=ax.transAxes)
        return

    X = np.array(all_vecs)

    if use_tsne:
        try:
            from sklearn.manifold import TSNE
            proj = TSNE(n_components=2, perplexity=min(30, len(X)-1),
                        random_state=42, metric="hamming").fit_transform(X)
            title = "T-SNE (Hamming)"
        except ImportError:
            from sklearn.decomposition import PCA
            proj = PCA(n_components=2).fit_transform(X)
            title = "PCA (sklearn t-SNE unavailable)"
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        proj = pca.fit_transform(X)
        var = pca.explained_variance_ratio_
        title = f"PCA ({var[0]:.0%} + {var[1]:.0%} variance)"

    # Color by round
    rounds = sorted(set(r for r, _ in labels))
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=min(rounds), vmax=max(rounds))

    # Get unique models for marker shapes
    models = sorted(set(m for _, m in labels))
    markers = ["o", "s", "^", "D", "v", "P", "*"][:len(models)]
    model_marker = {m: markers[i % len(markers)] for i, m in enumerate(models)}

    for i, (r, m) in enumerate(labels):
        ax.scatter(proj[i, 0], proj[i, 1],
                   c=[cmap(norm(r))], marker=model_marker[m],
                   s=40, alpha=0.6, edgecolors="white", linewidths=0.3)

    # Plot meta-godform trajectory
    if meta_lattices and len(meta_lattices) >= 2:
        meta_X = np.array(meta_lattices)
        if use_tsne:
            # Re-project with all data
            combined = np.vstack([X, meta_X])
            try:
                from sklearn.manifold import TSNE
                combined_proj = TSNE(n_components=2, perplexity=min(30, len(combined)-1),
                                     random_state=42, metric="hamming").fit_transform(combined)
                meta_proj = combined_proj[len(X):]
            except ImportError:
                meta_proj = PCA(n_components=2).fit_transform(meta_X)
        else:
            meta_proj = pca.transform(meta_X)

        # Draw trajectory line
        ax.plot(meta_proj[:, 0], meta_proj[:, 1], "k-", linewidth=2, alpha=0.8, zorder=5)
        for j in range(len(meta_proj)):
            ax.scatter(meta_proj[j, 0], meta_proj[j, 1],
                       c=[cmap(norm(j + 1))], marker="H", s=200,
                       edgecolors="black", linewidths=1.5, zorder=6)
            ax.annotate(f"R{j+1}", (meta_proj[j, 0], meta_proj[j, 1]),
                        fontsize=7, fontweight="bold", ha="center", va="bottom",
                        xytext=(0, 8), textcoords="offset points")

    # Legend for models
    for m in models:
        ax.scatter([], [], marker=model_marker[m], c="gray", s=40, label=m)
    ax.legend(fontsize=6, loc="upper right", ncol=2)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 2: Bit-flip heatmap (96 bits × rounds)
# ═══════════════════════════════════════════════════════════════════════

def plot_bit_heatmap(run: dict, ax: plt.Axes):
    """Heatmap of meta-godform bits across rounds."""
    axes = get_axes(run)
    trajectory = run["trajectory"]
    signs = []
    for rd in trajectory:
        sign = rd.get("godform_sign", rd.get("meta_godform_unicode", ""))
        if sign:
            signs.append(braille_to_bits(sign))

    if len(signs) < 2:
        ax.text(0.5, 0.5, "Need ≥2 rounds", ha="center", va="center",
                transform=ax.transAxes)
        return

    bit_matrix = np.array(signs)

    # Show the bit values
    im = ax.imshow(bit_matrix, aspect="auto", cmap="RdYlGn",
                   interpolation="nearest", vmin=0, vmax=1)

    # Mark flips
    for r in range(1, len(signs)):
        flips = np.where(bit_matrix[r] != bit_matrix[r-1])[0]
        for b in flips:
            ax.plot(b, r, "kx", markersize=3, markeredgewidth=0.8)

    # Axis labels
    ax.set_yticks(range(len(signs)))
    ax.set_yticklabels([f"R{i+1}" for i in range(len(signs))], fontsize=8)
    ax.set_xlabel(f"Bit position ({len(axes)} axes × 8 bits)", fontsize=8)

    # Axis boundaries
    for i in range(1, len(axes)):
        ax.axvline(i * 8 - 0.5, color="gray", linewidth=0.3, alpha=0.5)

    # Axis names at top
    for i, axis in enumerate(axes):
        ax.text(i * 8 + 3.5, -0.7, axis[:4], fontsize=5, ha="center",
                rotation=45, va="bottom")

    ax.set_title("Lattice Bit Evolution (× = flip)", fontsize=10, fontweight="bold")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 3: Axis evolution radar
# ═══════════════════════════════════════════════════════════════════════

def plot_axis_radar(run: dict, ax: plt.Axes):
    """Radar chart of axis values across rounds."""
    axes = get_axes(run)
    trajectory = run["trajectory"]
    n_axes = len(axes)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    cmap = plt.cm.viridis
    n_rounds = len(trajectory)

    for i, rd in enumerate(trajectory):
        vec = rd.get("meta_godform_vector", {})
        values = [vec.get(a, 0) for a in axes]
        values += values[:1]
        color = cmap(i / max(n_rounds - 1, 1))
        ax.plot(angles, values, "-", color=color, linewidth=1.5,
                label=f"R{rd['round']}", alpha=0.8)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([a[:5] for a in axes], fontsize=6)
    ax.set_ylim(0, 0.6)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels(["0.1", "0.2", "0.3", "0.4", "0.5"], fontsize=5)
    ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Axis Evolution", fontsize=10, fontweight="bold", pad=20)


# ═══════════════════════════════════════════════════════════════════════
# PLOT 4: Convergence timeline (entropy + bit flips)
# ═══════════════════════════════════════════════════════════════════════

def plot_convergence(run: dict, ax: plt.Axes):
    """Dual-axis plot: bit flips (left) and entropy (right) over rounds."""
    n_bits = get_n_bits(run)
    trajectory = run["trajectory"]
    rounds = [rd["round"] for rd in trajectory]
    entropy = [rd.get("mean_theological_entropy", 0) for rd in trajectory]
    flips = [rd.get("bit_flips") for rd in trajectory]

    # Entropy line
    color1 = "#2196F3"
    ax.plot(rounds, entropy, "o-", color=color1, linewidth=2, markersize=6, label="Entropy")
    ax.set_xlabel("Round", fontsize=9)
    ax.set_ylabel("Entropy (bits)", color=color1, fontsize=9)
    ax.tick_params(axis="y", labelcolor=color1)
    ax.set_ylim(0, n_bits)
    ax.axhline(n_bits / 2, color=color1, linestyle=":", alpha=0.3, label="Max disorder")

    # Bit flips on secondary axis
    ax2 = ax.twinx()
    color2 = "#FF5722"
    valid_flips = [(r, f) for r, f in zip(rounds, flips) if f is not None]
    if valid_flips:
        flip_rounds, flip_vals = zip(*valid_flips)
        ax2.bar(flip_rounds, flip_vals, color=color2, alpha=0.4, width=0.4, label="Bit flips")
        ax2.set_ylabel("Bit Flips", color=color2, fontsize=9)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(0, max(flip_vals) * 1.5)

    # Alpha annotation
    final = trajectory[-1]
    alpha = final.get("agreement", {}).get("alpha", None)
    if alpha is not None:
        ax.text(0.98, 0.02, f"α = {alpha:.3f}", transform=ax.transAxes,
                fontsize=9, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    ax.set_title("Convergence Timeline", fontsize=10, fontweight="bold")
    ax.set_xticks(rounds)


# ═══════════════════════════════════════════════════════════════════════
# PLOT 5: Sign trajectory (braille characters with Hamming distances)
# ═══════════════════════════════════════════════════════════════════════

def plot_sign_trajectory(run: dict, ax: plt.Axes):
    """Show the braille sign evolution with Hamming distances."""
    trajectory = run["trajectory"]
    signs = []
    for rd in trajectory:
        sign = rd.get("godform_sign", rd.get("meta_godform_unicode", ""))
        signs.append((rd["round"], sign))

    ax.set_xlim(-0.5, len(signs) - 0.5)
    ax.set_ylim(-1, 3)
    ax.axis("off")

    for i, (r, sign) in enumerate(signs):
        # Braille sign
        ax.text(i, 2, sign, fontsize=14, ha="center", va="center",
                fontfamily="monospace", fontweight="bold")
        ax.text(i, 1.2, f"Round {r}", fontsize=7, ha="center", va="center")

        # Hamming distance arrow
        if i > 0:
            prev_bits = braille_to_bits(signs[i-1][1])
            curr_bits = braille_to_bits(sign)
            hamming = int(np.sum(prev_bits != curr_bits))
            ax.annotate("", xy=(i - 0.1, 2), xytext=(i - 0.9, 2),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
            ax.text(i - 0.5, 2.5, f"{hamming} flips", fontsize=6,
                    ha="center", color="red")

    # Top axes for final round
    final = trajectory[-1]
    top = final.get("meta_godform_top_axes", [])
    vec = final.get("meta_godform_vector", {})
    top_str = ", ".join(f"{a}={vec.get(a, 0):.2f}" for a in top[:3])
    ax.text(len(signs) / 2 - 0.5, 0.2, f"Final: {top_str}", fontsize=8,
            ha="center", va="center", style="italic")

    ax.set_title("Godform Sign Trajectory", fontsize=10, fontweight="bold")


# ═══════════════════════════════════════════════════════════════════════
# MAIN: Compose all plots
# ═══════════════════════════════════════════════════════════════════════

def visualize_run(run_path: str, output_path: Optional[str] = None):
    """Generate the full visualization dashboard for a Godform run."""
    run = load_run(run_path)
    run_name = Path(run_path).stem

    domain_label = run.get("domain_label", "Godform")
    domain_name = run.get("domain", "theology")
    n_bits = get_n_bits(run)

    fig = plt.figure(figsize=(18, 14), facecolor="white")
    fig.suptitle(f"{domain_label} Braiding ({domain_name}, {n_bits}-bit): {run_name}\n"
                 f"{run['n_models']} models × {run['n_prompts']} prompts × "
                 f"{run['n_rounds']} rounds",
                 fontsize=13, fontweight="bold", y=0.98)

    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35,
                  top=0.92, bottom=0.05, left=0.06, right=0.96)

    # Row 1: PCA, T-SNE, Sign trajectory
    ax_pca = fig.add_subplot(gs[0, 0])
    plot_pca_trajectory(run, ax_pca, use_tsne=False)

    ax_tsne = fig.add_subplot(gs[0, 1])
    plot_pca_trajectory(run, ax_tsne, use_tsne=True)

    ax_sign = fig.add_subplot(gs[0, 2])
    plot_sign_trajectory(run, ax_sign)

    # Row 2: Bit heatmap (spans 2 cols), Convergence
    ax_heat = fig.add_subplot(gs[1, :2])
    plot_bit_heatmap(run, ax_heat)

    ax_conv = fig.add_subplot(gs[1, 2])
    plot_convergence(run, ax_conv)

    # Row 3: Radar (polar), Prestige evolution, Agreement per axis
    ax_radar = fig.add_subplot(gs[2, 0], polar=True)
    plot_axis_radar(run, ax_radar)

    # Prestige evolution
    ax_prest = fig.add_subplot(gs[2, 1])
    plot_prestige_evolution(run, ax_prest)

    # Per-axis agreement
    ax_agree = fig.add_subplot(gs[2, 2])
    plot_axis_agreement(run, ax_agree)

    if output_path is None:
        output_path = str(Path(run_path).with_suffix(".png"))

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)
    return output_path


def plot_prestige_evolution(run: dict, ax: plt.Axes):
    """Bar chart of prestige weights per model across rounds."""
    trajectory = run["trajectory"]
    models = sorted(set(
        m for rd in trajectory for m in rd.get("prestige", {}).keys()
    ))

    if not models:
        ax.text(0.5, 0.5, "No prestige data", ha="center", va="center",
                transform=ax.transAxes)
        return

    n_rounds = len(trajectory)
    x = np.arange(len(models))
    width = 0.8 / n_rounds
    cmap = plt.cm.viridis

    for i, rd in enumerate(trajectory):
        prestige = rd.get("prestige", {})
        vals = [prestige.get(m, 0) for m in models]
        color = cmap(i / max(n_rounds - 1, 1))
        ax.bar(x + i * width - 0.4 + width/2, vals, width,
               color=color, alpha=0.8, label=f"R{rd['round']}")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=6, rotation=30, ha="right")
    ax.set_ylabel("Prestige Weight", fontsize=8)
    ax.set_ylim(0.8, 1.05)
    ax.legend(fontsize=5, ncol=n_rounds)
    ax.set_title("Prestige Evolution", fontsize=10, fontweight="bold")


def plot_axis_agreement(run: dict, ax: plt.Axes):
    """Horizontal bar chart of per-axis Krippendorff's alpha (final round)."""
    final = run["trajectory"][-1]
    per_axis = final.get("agreement", {}).get("per_axis", {})

    if not per_axis:
        ax.text(0.5, 0.5, "No per-axis agreement data", ha="center",
                va="center", transform=ax.transAxes)
        return

    axes_sorted = sorted(per_axis.keys(), key=lambda a: per_axis[a])
    vals = [per_axis[a] for a in axes_sorted]

    colors = ["#f44336" if v < 0.4 else "#FF9800" if v < 0.667 else "#4CAF50"
              for v in vals]

    ax.barh(range(len(axes_sorted)), vals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(axes_sorted)))
    ax.set_yticklabels(axes_sorted, fontsize=7)
    ax.set_xlabel("Krippendorff's α", fontsize=8)
    ax.axvline(0.667, color="orange", linestyle="--", alpha=0.5, label="Acceptable")
    ax.axvline(0.8, color="green", linestyle="--", alpha=0.5, label="Good")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=6)
    ax.set_title("Per-Axis Agreement (Final)", fontsize=10, fontweight="bold")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python godform/visualize.py <run.json> [output.png]")
        sys.exit(1)

    run_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    visualize_run(run_path, output_path)
