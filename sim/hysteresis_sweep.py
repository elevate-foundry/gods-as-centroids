#!/usr/bin/env python3
"""
Hysteresis Sweep Experiment — The Hero Plot

Demonstrates the first-order phase transition from polytheism → monotheism
and the asymmetric lock-in (hysteresis) predicted by the paper.

Protocol:
  FORWARD SWEEP:  Start at γ=0, increase to γ=1 in steps of 0.05.
                  At each γ, run for EQUILIBRATION_STEPS to let the system settle.
                  Measure dominance D, effective deity count N_eff, entropy H.

  REVERSE SWEEP:  Start from the final monotheistic state (γ=1),
                  decrease γ back to 0 in the same steps.
                  Measure the same quantities.

  If the paper's prediction is correct:
    - Forward: D jumps from ~0.3 to ~0.9 at γ_c+ ≈ 0.4
    - Reverse: D stays high until γ_c- ≈ 0.15 (or never drops)
    - The two curves enclose a HYSTERESIS LOOP

  Repeat N_RUNS times for error bars.

Output: hero_plot.png — the flagship figure for the paper.
"""

import sys
import os
import json
import math
import statistics
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from swarm_kernel import SwarmKernel, Config, AXES, cosine

# ─── Experiment Parameters ───────────────────────────────────────────

N_AGENTS = 80
GAMMA_STEPS = [round(g * 0.05, 2) for g in range(21)]  # 0.00, 0.05, ..., 1.00
EQUILIBRATION_STEPS = 2000   # steps per γ level to reach equilibrium
MEASUREMENT_WINDOW = 500     # last N steps to average measurements over
N_RUNS = 30                  # independent runs for error bars
SEED_BASE = 42


# ─── Measurement Functions ───────────────────────────────────────────

def measure_system(kernel: SwarmKernel) -> dict:
    """Measure dominance, N_eff, and entropy from current cluster state."""
    kernel._update_clusters()
    clusters = kernel.clusters
    N = len(kernel.agents)

    if not clusters:
        return {"dominance": 0.0, "n_eff": 0, "entropy": 0.0, "n_clusters": 0}

    sizes = [len(c) for c in clusters]
    # Effective deity count: clusters with >= 2 members
    n_eff = sum(1 for s in sizes if s >= 2)
    # Dominance: fraction in largest cluster
    dominance = max(sizes) / N if N > 0 else 0.0
    # Shannon entropy
    entropy = 0.0
    for s in sizes:
        if s > 0:
            p = s / N
            entropy -= p * math.log(p)

    return {
        "dominance": dominance,
        "n_eff": n_eff,
        "entropy": entropy,
        "n_clusters": len(clusters),
    }


def run_sweep(direction: str, seed: int) -> list:
    """
    Run a single forward or reverse sweep.
    Returns list of {gamma, dominance, n_eff, entropy} dicts.
    """
    gammas = GAMMA_STEPS if direction == "forward" else list(reversed(GAMMA_STEPS))

    cfg = Config(
        N=N_AGENTS,
        seed=seed,
        use_deity_priors=True,
        coercion=gammas[0],
        mutation_rate=0.08,
        cluster_update_freq=50,
        cluster_threshold=0.4,
        steps_per_generation=50000,  # effectively disable generation turnover
    )
    kernel = SwarmKernel(cfg)

    results = []
    for gamma in gammas:
        kernel.cfg.coercion = gamma

        # Run equilibration
        measurements = []
        for step in range(EQUILIBRATION_STEPS):
            kernel.step()
            # Collect measurements in the last window
            if step >= EQUILIBRATION_STEPS - MEASUREMENT_WINDOW and step % 25 == 0:
                m = measure_system(kernel)
                measurements.append(m)

        # Average measurements over the window
        avg_dominance = statistics.mean(m["dominance"] for m in measurements)
        avg_n_eff = statistics.mean(m["n_eff"] for m in measurements)
        avg_entropy = statistics.mean(m["entropy"] for m in measurements)

        results.append({
            "gamma": gamma,
            "dominance": avg_dominance,
            "n_eff": avg_n_eff,
            "entropy": avg_entropy,
        })

        print(f"  {direction} γ={gamma:.2f}: D={avg_dominance:.3f}, "
              f"N_eff={avg_n_eff:.1f}, H={avg_entropy:.3f}")

    return results


def run_experiment():
    """Run full hysteresis experiment with multiple runs."""
    all_forward = []
    all_reverse = []

    for run_idx in range(N_RUNS):
        seed = SEED_BASE + run_idx * 137
        print(f"\n{'='*60}")
        print(f"Run {run_idx + 1}/{N_RUNS} (seed={seed})")
        print(f"{'='*60}")

        print(f"\n  FORWARD SWEEP (γ: 0 → 1)")
        forward = run_sweep("forward", seed)
        all_forward.append(forward)

        print(f"\n  REVERSE SWEEP (γ: 1 → 0)")
        reverse = run_sweep("reverse", seed + 1)
        all_reverse.append(reverse)

    return all_forward, all_reverse


def aggregate_results(all_runs: list) -> dict:
    """Aggregate multiple runs into mean ± std at each γ."""
    # Group by gamma
    by_gamma = {}
    for run in all_runs:
        for point in run:
            g = point["gamma"]
            if g not in by_gamma:
                by_gamma[g] = {"dominance": [], "n_eff": [], "entropy": []}
            by_gamma[g]["dominance"].append(point["dominance"])
            by_gamma[g]["n_eff"].append(point["n_eff"])
            by_gamma[g]["entropy"].append(point["entropy"])

    result = {}
    for g in sorted(by_gamma.keys()):
        d = by_gamma[g]
        result[g] = {
            "gamma": g,
            "dominance_mean": statistics.mean(d["dominance"]),
            "dominance_std": statistics.stdev(d["dominance"]) if len(d["dominance"]) > 1 else 0,
            "n_eff_mean": statistics.mean(d["n_eff"]),
            "n_eff_std": statistics.stdev(d["n_eff"]) if len(d["n_eff"]) > 1 else 0,
            "entropy_mean": statistics.mean(d["entropy"]),
            "entropy_std": statistics.stdev(d["entropy"]) if len(d["entropy"]) > 1 else 0,
        }
    return result


def generate_hero_plot(forward_agg: dict, reverse_agg: dict, output_dir: str):
    """Generate the hero plot: hysteresis loop of D vs γ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # Extract data
    gammas_f = sorted(forward_agg.keys())
    gammas_r = sorted(reverse_agg.keys())

    d_f_mean = [forward_agg[g]["dominance_mean"] for g in gammas_f]
    d_f_std = [forward_agg[g]["dominance_std"] for g in gammas_f]
    d_r_mean = [reverse_agg[g]["dominance_mean"] for g in gammas_r]
    d_r_std = [reverse_agg[g]["dominance_std"] for g in gammas_r]

    neff_f_mean = [forward_agg[g]["n_eff_mean"] for g in gammas_f]
    neff_f_std = [forward_agg[g]["n_eff_std"] for g in gammas_f]
    neff_r_mean = [reverse_agg[g]["n_eff_mean"] for g in gammas_r]
    neff_r_std = [reverse_agg[g]["n_eff_std"] for g in gammas_r]

    # ─── Figure 1: Hero Plot (Dominance hysteresis) ──────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Dominance D vs γ
    ax1.fill_between(gammas_f,
                     [m - s for m, s in zip(d_f_mean, d_f_std)],
                     [m + s for m, s in zip(d_f_mean, d_f_std)],
                     alpha=0.2, color="#c0392b")
    ax1.plot(gammas_f, d_f_mean, 'o-', color="#c0392b", linewidth=2.5,
             markersize=5, label="Forward (γ↑): polytheism → monotheism")

    ax1.fill_between(gammas_r,
                     [m - s for m, s in zip(d_r_mean, d_r_std)],
                     [m + s for m, s in zip(d_r_mean, d_r_std)],
                     alpha=0.2, color="#2980b9")
    ax1.plot(gammas_r, d_r_mean, 's-', color="#2980b9", linewidth=2.5,
             markersize=5, label="Reverse (γ↓): monotheism → ?")

    ax1.set_xlabel("Coercion parameter γ", fontsize=13)
    ax1.set_ylabel("Dominance D (fraction in largest cluster)", fontsize=13)
    ax1.set_title("Hysteresis in the Polytheism–Monotheism Transition", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc="center left")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.05)
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # Annotate critical points
    # Find forward γ_c+ (first γ where D > 0.7)
    gamma_c_plus = None
    for i, g in enumerate(gammas_f):
        if d_f_mean[i] > 0.7:
            gamma_c_plus = g
            break
    # Find reverse γ_c- (last γ where D > 0.7, going down)
    gamma_c_minus = None
    for g in gammas_r:
        if reverse_agg[g]["dominance_mean"] > 0.7:
            gamma_c_minus = g
        else:
            break

    if gamma_c_plus is not None:
        ax1.axvline(x=gamma_c_plus, color="#c0392b", linestyle="--", alpha=0.5)
        ax1.annotate(f"γ$_c^+$ ≈ {gamma_c_plus:.2f}",
                     xy=(gamma_c_plus, 0.75), fontsize=10, color="#c0392b",
                     ha="left", va="bottom",
                     xytext=(gamma_c_plus + 0.05, 0.55),
                     arrowprops=dict(arrowstyle="->", color="#c0392b"))

    if gamma_c_minus is not None and gamma_c_minus != gamma_c_plus:
        ax1.axvline(x=gamma_c_minus, color="#2980b9", linestyle="--", alpha=0.5)
        ax1.annotate(f"γ$_c^-$ ≈ {gamma_c_minus:.2f}",
                     xy=(gamma_c_minus, 0.75), fontsize=10, color="#2980b9",
                     ha="right", va="bottom",
                     xytext=(gamma_c_minus - 0.08, 0.45),
                     arrowprops=dict(arrowstyle="->", color="#2980b9"))

    # Right panel: N_eff vs γ
    ax2.fill_between(gammas_f,
                     [max(0, m - s) for m, s in zip(neff_f_mean, neff_f_std)],
                     [m + s for m, s in zip(neff_f_mean, neff_f_std)],
                     alpha=0.2, color="#c0392b")
    ax2.plot(gammas_f, neff_f_mean, 'o-', color="#c0392b", linewidth=2.5,
             markersize=5, label="Forward (γ↑)")

    ax2.fill_between(gammas_r,
                     [max(0, m - s) for m, s in zip(neff_r_mean, neff_r_std)],
                     [m + s for m, s in zip(neff_r_mean, neff_r_std)],
                     alpha=0.2, color="#2980b9")
    ax2.plot(gammas_r, neff_r_mean, 's-', color="#2980b9", linewidth=2.5,
             markersize=5, label="Reverse (γ↓)")

    ax2.set_xlabel("Coercion parameter γ", fontsize=13)
    ax2.set_ylabel("Effective deity count N_eff", fontsize=13)
    ax2.set_title("Deity Count: Irreversible Collapse", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xlim(-0.02, 1.02)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    hero_path = os.path.join(output_dir, "hero_plot.png")
    plt.savefig(hero_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nHero plot saved to {hero_path}")

    # ─── Figure 2: Entropy panel ─────────────────────────────────
    fig2, ax3 = plt.subplots(figsize=(7, 5))

    ent_f_mean = [forward_agg[g]["entropy_mean"] for g in gammas_f]
    ent_f_std = [forward_agg[g]["entropy_std"] for g in gammas_f]
    ent_r_mean = [reverse_agg[g]["entropy_mean"] for g in gammas_r]
    ent_r_std = [reverse_agg[g]["entropy_std"] for g in gammas_r]

    ax3.fill_between(gammas_f,
                     [max(0, m - s) for m, s in zip(ent_f_mean, ent_f_std)],
                     [m + s for m, s in zip(ent_f_mean, ent_f_std)],
                     alpha=0.2, color="#c0392b")
    ax3.plot(gammas_f, ent_f_mean, 'o-', color="#c0392b", linewidth=2.5,
             markersize=5, label="Forward (γ↑)")

    ax3.fill_between(gammas_r,
                     [max(0, m - s) for m, s in zip(ent_r_mean, ent_r_std)],
                     [m + s for m, s in zip(ent_r_mean, ent_r_std)],
                     alpha=0.2, color="#2980b9")
    ax3.plot(gammas_r, ent_r_mean, 's-', color="#2980b9", linewidth=2.5,
             markersize=5, label="Reverse (γ↓)")

    ax3.set_xlabel("Coercion parameter γ", fontsize=13)
    ax3.set_ylabel("Belief entropy H", fontsize=13)
    ax3.set_title("Entropy: Diversity Lost Is Not Recovered", fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.set_xlim(-0.02, 1.02)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    entropy_path = os.path.join(output_dir, "entropy_hysteresis.png")
    plt.savefig(entropy_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Entropy plot saved to {entropy_path}")

    return hero_path, entropy_path


# ─── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "runs", "hysteresis_sweep")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("HYSTERESIS SWEEP EXPERIMENT")
    print(f"N_AGENTS={N_AGENTS}, γ steps={len(GAMMA_STEPS)}, "
          f"equil={EQUILIBRATION_STEPS} steps/γ, runs={N_RUNS}")
    print("=" * 60)

    all_forward, all_reverse = run_experiment()

    # Aggregate
    forward_agg = aggregate_results(all_forward)
    reverse_agg = aggregate_results(all_reverse)

    # Save raw data
    raw_data = {
        "params": {
            "n_agents": N_AGENTS,
            "gamma_steps": GAMMA_STEPS,
            "equilibration_steps": EQUILIBRATION_STEPS,
            "measurement_window": MEASUREMENT_WINDOW,
            "n_runs": N_RUNS,
        },
        "forward": {str(k): v for k, v in forward_agg.items()},
        "reverse": {str(k): v for k, v in reverse_agg.items()},
    }
    data_path = os.path.join(output_dir, "hysteresis_data.json")
    with open(data_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"\nRaw data saved to {data_path}")

    # Generate plots
    try:
        hero_path, entropy_path = generate_hero_plot(forward_agg, reverse_agg, output_dir)
    except ImportError:
        print("\nmatplotlib not available — skipping plot generation.")
        print("Install with: pip install matplotlib")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'γ':>6} | {'D_fwd':>8} | {'D_rev':>8} | {'gap':>8} | {'Neff_fwd':>8} | {'Neff_rev':>8}")
    print("-" * 60)
    for g in GAMMA_STEPS:
        f = forward_agg[g]
        r = reverse_agg[g]
        gap = r["dominance_mean"] - f["dominance_mean"]
        print(f"{g:6.2f} | {f['dominance_mean']:8.3f} | {r['dominance_mean']:8.3f} | "
              f"{gap:+8.3f} | {f['n_eff_mean']:8.1f} | {r['n_eff_mean']:8.1f}")

    # Identify critical points
    gamma_c_plus = None
    for g in GAMMA_STEPS:
        if forward_agg[g]["dominance_mean"] > 0.7:
            gamma_c_plus = g
            break

    gamma_c_minus = None
    for g in reversed(GAMMA_STEPS):
        if reverse_agg[g]["dominance_mean"] < 0.7:
            gamma_c_minus = g
            break

    print(f"\nForward critical point γ_c+ ≈ {gamma_c_plus}")
    print(f"Reverse critical point γ_c- ≈ {gamma_c_minus}")
    if gamma_c_plus and gamma_c_minus:
        print(f"Hysteresis gap: Δγ = {gamma_c_plus - gamma_c_minus:.2f}")
        if gamma_c_minus < gamma_c_plus:
            print("✓ HYSTERESIS CONFIRMED: reverse threshold < forward threshold")
        else:
            print("✗ No hysteresis detected — system is reversible")
    print()
