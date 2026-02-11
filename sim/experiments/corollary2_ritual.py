#!/usr/bin/env python3
"""
Corollary 2 (Ritual Stabilization) — Reproduction Script
=========================================================
Paper claim (§5.2): "With ritual bonus r=0.15 and period T=50, centroid drift
rate decreases by 40% compared to r=0 (measured as mean centroid displacement
per 1,000 steps)."

This script runs the GABM with and without ritual bonus across multiple
replicates and measures centroid drift rate, producing saved JSON results
and a summary plot.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import math
import random
import statistics
import time
from collections import defaultdict

from swarm_kernel import SwarmKernel, Config, AXES, cosine, norm

# ── Parameters ──────────────────────────────────────────────────────

N_AGENTS = 80
STEPS = 5000
MEASURE_INTERVAL = 100       # measure centroid positions every N steps
N_REPLICATES = 30
RITUAL_PERIOD = 50
RITUAL_BONUS_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
BASE_SEED = 42

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'runs', 'corollary2_ritual')


def centroid_drift_rate(centroid_history, interval):
    """Compute mean centroid displacement per 1000 steps.
    
    centroid_history: list of (step, [centroid_dicts]) snapshots
    Returns mean cosine displacement per 1000 steps across all tracked centroids.
    """
    if len(centroid_history) < 2:
        return 0.0

    displacements = []
    for i in range(1, len(centroid_history)):
        t_prev, centroids_prev = centroid_history[i - 1]
        t_curr, centroids_curr = centroid_history[i]
        dt = t_curr - t_prev
        if dt == 0:
            continue

        # Match centroids by nearest cosine similarity
        for cp in centroids_prev:
            best_dist = float('inf')
            for cc in centroids_curr:
                d = 1.0 - cosine(cp, cc)
                if d < best_dist:
                    best_dist = d
            displacements.append(best_dist / dt * 1000)  # normalize to per-1000-steps

    return statistics.mean(displacements) if displacements else 0.0


def run_single(ritual_bonus, seed):
    """Run one simulation and return centroid drift rate."""
    cfg = Config(
        N=N_AGENTS,
        steps_per_generation=STEPS,
        ritual_bonus=ritual_bonus,
        ritual_period=RITUAL_PERIOD,
        seed=seed,
        coercion=0.1,          # mild coercion so clusters form
        use_deity_priors=True,
        cluster_update_freq=50,
    )
    kernel = SwarmKernel(cfg)

    centroid_history = []
    for step in range(1, STEPS + 1):
        kernel.transmit()
        if step % MEASURE_INTERVAL == 0 and kernel.centroids:
            snapshot = [c.copy() for c in kernel.centroids]
            centroid_history.append((step, snapshot))

    drift = centroid_drift_rate(centroid_history, MEASURE_INTERVAL)
    n_eff = len(kernel.centroids)
    return drift, n_eff


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("Corollary 2: Ritual Stabilization (§5.2)")
    print("=" * 60)

    all_results = {}
    for rb in RITUAL_BONUS_VALUES:
        print(f"\n  ritual_bonus = {rb:.2f}  ({N_REPLICATES} replicates)")
        drifts = []
        n_effs = []
        for rep in range(N_REPLICATES):
            drift, n_eff = run_single(rb, BASE_SEED + rep)
            drifts.append(drift)
            n_effs.append(n_eff)
            if (rep + 1) % 10 == 0:
                print(f"    rep {rep+1}/{N_REPLICATES}: drift={drift:.4f}")

        mean_drift = statistics.mean(drifts)
        std_drift = statistics.stdev(drifts) if len(drifts) > 1 else 0.0
        mean_neff = statistics.mean(n_effs)

        all_results[str(rb)] = {
            'ritual_bonus': rb,
            'mean_drift': mean_drift,
            'std_drift': std_drift,
            'mean_neff': mean_neff,
            'drifts': drifts,
            'n_effs': n_effs,
        }
        print(f"    → mean drift = {mean_drift:.4f} ± {std_drift:.4f}, N_eff = {mean_neff:.1f}")

    # ── Compute reduction relative to r=0 baseline ──
    baseline_drift = all_results['0.0']['mean_drift']
    for key, res in all_results.items():
        if baseline_drift > 0:
            res['reduction_pct'] = (1.0 - res['mean_drift'] / baseline_drift) * 100
        else:
            res['reduction_pct'] = 0.0

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Ritual Bonus':>14} {'Mean Drift':>12} {'Std':>10} {'Reduction':>12}")
    for key in sorted(all_results.keys(), key=float):
        r = all_results[key]
        print(f"  r = {r['ritual_bonus']:.2f}     {r['mean_drift']:10.4f}   {r['std_drift']:8.4f}   {r['reduction_pct']:8.1f}%")

    target = all_results.get('0.15', {})
    if target:
        print(f"\nPaper claim: ~40% reduction at r=0.15")
        print(f"Measured:    {target['reduction_pct']:.1f}% reduction at r=0.15")

    # ── Save ──
    output = {
        'experiment': 'corollary2_ritual_stabilization',
        'paper_section': '§5.2',
        'paper_claim': 'Centroid drift rate decreases by ~40% with ritual bonus r=0.15, T=50',
        'parameters': {
            'N': N_AGENTS,
            'steps': STEPS,
            'measure_interval': MEASURE_INTERVAL,
            'n_replicates': N_REPLICATES,
            'ritual_period': RITUAL_PERIOD,
            'ritual_bonus_values': RITUAL_BONUS_VALUES,
        },
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'drifts' and kk != 'n_effs'}
                    for k, v in all_results.items()},
        'full_results': all_results,
        'baseline_drift': baseline_drift,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    json_path = os.path.join(OUTPUT_DIR, 'corollary2_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        bonuses = [all_results[str(rb)]['ritual_bonus'] for rb in RITUAL_BONUS_VALUES]
        means = [all_results[str(rb)]['mean_drift'] for rb in RITUAL_BONUS_VALUES]
        stds = [all_results[str(rb)]['std_drift'] for rb in RITUAL_BONUS_VALUES]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.errorbar(bonuses, means, yerr=stds, fmt='o-', capsize=5, color='#2196F3', linewidth=2)
        ax1.axhline(y=baseline_drift * 0.6, color='red', linestyle='--', alpha=0.5, label='40% reduction target')
        ax1.set_xlabel('Ritual Bonus (r)', fontsize=12)
        ax1.set_ylabel('Mean Centroid Drift Rate\n(cosine displacement per 1000 steps)', fontsize=11)
        ax1.set_title('Corollary 2: Ritual Stabilization', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        reductions = [all_results[str(rb)]['reduction_pct'] for rb in RITUAL_BONUS_VALUES]
        ax2.bar(bonuses, reductions, width=0.04, color='#4CAF50', alpha=0.8)
        ax2.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='Paper claim: 40%')
        ax2.set_xlabel('Ritual Bonus (r)', fontsize=12)
        ax2.set_ylabel('Drift Reduction (%)', fontsize=12)
        ax2.set_title('Reduction vs Baseline (r=0)', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, 'corollary2_plot.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved: {plot_path}")
    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == '__main__':
    main()
