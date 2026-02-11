#!/usr/bin/env python3
"""
Corollary 3 (Prestige Convergence) — Reproduction Script
=========================================================
Paper claim (§5.3): "At γ=0.3 (below γ_c for default α), increasing α from
0.2 to 0.5 reduces N_eff from 4.2 to 1.8 (averaged over 50 runs)."

This script sweeps prestige amplification α at fixed moderate coercion and
measures effective deity count, producing saved JSON results and a summary plot.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import statistics
import time

from swarm_kernel import SwarmKernel, Config

# ── Parameters ──────────────────────────────────────────────────────

N_AGENTS = 80
STEPS = 3000
COERCION = 0.3                # below γ_c for default α
N_REPLICATES = 50
ALPHA_VALUES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
BASE_SEED = 42

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'runs', 'corollary3_prestige')


def run_single(alpha, seed):
    """Run one simulation and return final N_eff and dominance."""
    cfg = Config(
        N=N_AGENTS,
        steps_per_generation=STEPS,
        prestige_alpha=alpha,
        coercion=COERCION,
        seed=seed,
        use_deity_priors=True,
        cluster_update_freq=50,
    )
    kernel = SwarmKernel(cfg)

    for _ in range(STEPS):
        kernel.transmit()

    n_eff = len(kernel.centroids)
    if kernel.clusters:
        max_cluster = max(len(c) for c in kernel.clusters)
        dominance = max_cluster / len(kernel.agents)
    else:
        dominance = 0.0

    return n_eff, dominance


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("Corollary 3: Prestige Convergence (§5.3)")
    print(f"Fixed coercion γ = {COERCION}, N = {N_AGENTS}, {N_REPLICATES} replicates")
    print("=" * 60)

    all_results = {}
    for alpha in ALPHA_VALUES:
        print(f"\n  α = {alpha:.2f}")
        n_effs = []
        dominances = []
        for rep in range(N_REPLICATES):
            n_eff, dom = run_single(alpha, BASE_SEED + rep)
            n_effs.append(n_eff)
            dominances.append(dom)
            if (rep + 1) % 10 == 0:
                print(f"    rep {rep+1}/{N_REPLICATES}: N_eff={n_eff}, D={dom:.3f}")

        mean_neff = statistics.mean(n_effs)
        std_neff = statistics.stdev(n_effs) if len(n_effs) > 1 else 0.0
        mean_dom = statistics.mean(dominances)
        std_dom = statistics.stdev(dominances) if len(dominances) > 1 else 0.0

        all_results[str(alpha)] = {
            'alpha': alpha,
            'mean_neff': mean_neff,
            'std_neff': std_neff,
            'mean_dominance': mean_dom,
            'std_dominance': std_dom,
            'n_effs': n_effs,
            'dominances': dominances,
        }
        print(f"    → N_eff = {mean_neff:.1f} ± {std_neff:.1f}, D = {mean_dom:.3f} ± {std_dom:.3f}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Alpha':>8} {'N_eff':>10} {'±':>6} {'Dominance':>12} {'±':>8}")
    for key in sorted(all_results.keys(), key=float):
        r = all_results[key]
        print(f"  {r['alpha']:.2f}    {r['mean_neff']:8.1f}  {r['std_neff']:5.1f}    {r['mean_dominance']:10.3f}  {r['std_dominance']:6.3f}")

    r02 = all_results.get('0.2', all_results.get('0.20', {}))
    r05 = all_results.get('0.5', all_results.get('0.50', {}))
    if r02 and r05:
        print(f"\nPaper claim: α=0.2 → N_eff≈4.2, α=0.5 → N_eff≈1.8")
        print(f"Measured:    α=0.2 → N_eff={r02['mean_neff']:.1f}, α=0.5 → N_eff={r05['mean_neff']:.1f}")

    # ── Save ──
    output = {
        'experiment': 'corollary3_prestige_convergence',
        'paper_section': '§5.3',
        'paper_claim': 'At γ=0.3, increasing α from 0.2 to 0.5 reduces N_eff from ~4.2 to ~1.8',
        'parameters': {
            'N': N_AGENTS,
            'steps': STEPS,
            'coercion': COERCION,
            'n_replicates': N_REPLICATES,
            'alpha_values': ALPHA_VALUES,
        },
        'results': {k: {kk: vv for kk, vv in v.items() if kk not in ('n_effs', 'dominances')}
                    for k, v in all_results.items()},
        'full_results': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    json_path = os.path.join(OUTPUT_DIR, 'corollary3_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        alphas = [all_results[str(a)]['alpha'] for a in ALPHA_VALUES]
        means = [all_results[str(a)]['mean_neff'] for a in ALPHA_VALUES]
        stds = [all_results[str(a)]['std_neff'] for a in ALPHA_VALUES]
        doms = [all_results[str(a)]['mean_dominance'] for a in ALPHA_VALUES]
        dom_stds = [all_results[str(a)]['std_dominance'] for a in ALPHA_VALUES]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.errorbar(alphas, means, yerr=stds, fmt='o-', capsize=5, color='#E91E63', linewidth=2)
        ax1.axhline(y=4.2, color='blue', linestyle='--', alpha=0.4, label='Paper: N_eff=4.2 at α=0.2')
        ax1.axhline(y=1.8, color='red', linestyle='--', alpha=0.4, label='Paper: N_eff=1.8 at α=0.5')
        ax1.set_xlabel('Prestige Amplification (α)', fontsize=12)
        ax1.set_ylabel('Effective Deity Count (N_eff)', fontsize=12)
        ax1.set_title(f'Corollary 3: Prestige Convergence (γ={COERCION})', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.errorbar(alphas, doms, yerr=dom_stds, fmt='s-', capsize=5, color='#FF9800', linewidth=2)
        ax2.set_xlabel('Prestige Amplification (α)', fontsize=12)
        ax2.set_ylabel('Dominance (D)', fontsize=12)
        ax2.set_title('Dominance vs Prestige', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, 'corollary3_plot.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved: {plot_path}")
    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == '__main__':
    main()
