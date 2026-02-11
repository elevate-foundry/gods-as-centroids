#!/usr/bin/env python3
"""
Historical Backtesting (§6.2) — Reproduction Script
====================================================
Paper claim: "The model's output (with manually scheduled coercion changes
matching known political events) achieves a Pearson correlation of r=0.82
with the historical N_eff series (p < 0.001, n = 23 epochs)."

This script runs the GABM with historically-scheduled coercion changes and
compares the resulting N_eff trajectory against historical estimates of
religious diversity from 3000 BCE to 2025 CE.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import math
import statistics
import time
from scipy import stats as scipy_stats

from swarm_kernel import SwarmKernel, Config

# ── Historical Data ─────────────────────────────────────────────────
# Sources: Pew Research, World Religion Database, Johnson & Grim (2013),
# Seshat Global History Databank. N_eff = effective number of major
# traditions each holding >3% of the relevant population.

HISTORICAL_EPOCHS = [
    {"year": -3000, "label": "Early Bronze Age",          "nEff": 15, "coercion": 0.05},
    {"year": -2500, "label": "Sumerian city-states",      "nEff": 12, "coercion": 0.10},
    {"year": -2000, "label": "Middle Bronze Age",         "nEff": 14, "coercion": 0.08},
    {"year": -1500, "label": "Late Bronze Age",           "nEff": 13, "coercion": 0.10},
    {"year": -1200, "label": "Bronze Age Collapse",       "nEff": 16, "coercion": 0.03},
    {"year": -800,  "label": "Early Iron Age",            "nEff": 14, "coercion": 0.08},
    {"year": -600,  "label": "Axial Age peak",            "nEff": 18, "coercion": 0.05},
    {"year": -400,  "label": "Classical Greece/Persia",   "nEff": 15, "coercion": 0.12},
    {"year": -200,  "label": "Hellenistic syncretism",    "nEff": 12, "coercion": 0.15},
    {"year": 0,     "label": "Roman Empire (early)",      "nEff": 10, "coercion": 0.20},
    {"year": 100,   "label": "Early Christianity",        "nEff": 10, "coercion": 0.18},
    {"year": 300,   "label": "Pre-Theodosian",            "nEff": 8,  "coercion": 0.30},
    {"year": 400,   "label": "Post-Theodosian",           "nEff": 4,  "coercion": 0.75},
    {"year": 600,   "label": "Early Islam",               "nEff": 5,  "coercion": 0.55},
    {"year": 800,   "label": "Carolingian/Abbasid",       "nEff": 4,  "coercion": 0.65},
    {"year": 1000,  "label": "High Middle Ages",          "nEff": 4,  "coercion": 0.70},
    {"year": 1200,  "label": "Crusades era",              "nEff": 4,  "coercion": 0.72},
    {"year": 1400,  "label": "Late Medieval",             "nEff": 4,  "coercion": 0.60},
    {"year": 1550,  "label": "Post-Reformation",          "nEff": 6,  "coercion": 0.50},
    {"year": 1700,  "label": "Enlightenment",             "nEff": 6,  "coercion": 0.40},
    {"year": 1850,  "label": "Industrial era",            "nEff": 7,  "coercion": 0.30},
    {"year": 1950,  "label": "Post-WWII",                 "nEff": 7,  "coercion": 0.20},
    {"year": 2025,  "label": "Modern",                    "nEff": 7,  "coercion": 0.12},
]

N_AGENTS = 80
STEPS_PER_EPOCH = 200         # 1 step ≈ 1 year, each epoch ~200 years
N_REPLICATES = 30
BASE_SEED = 42
PROPHET_EPOCHS = {-600, 0, 100, 600, 1550}  # Axial sages, Jesus, Muhammad, Luther

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'runs', 'backtesting')


def run_single(seed):
    """Run one historical trajectory and return simulated N_eff per epoch."""
    cfg = Config(
        N=N_AGENTS,
        steps_per_generation=STEPS_PER_EPOCH,
        seed=seed,
        coercion=HISTORICAL_EPOCHS[0]['coercion'],
        use_deity_priors=True,
        cluster_update_freq=50,
        mutation_rate=0.08,
        prophet_rate=0.0,
    )
    kernel = SwarmKernel(cfg)

    sim_neffs = []
    for epoch in HISTORICAL_EPOCHS:
        # Update coercion for this epoch
        kernel.cfg.coercion = epoch['coercion']

        # Enable prophets for specific epochs
        if epoch['year'] in PROPHET_EPOCHS:
            kernel.cfg.prophet_rate = 0.005
        else:
            kernel.cfg.prophet_rate = 0.0

        # Reformation-era fission: increase mutation
        if epoch['year'] == 1550:
            kernel.cfg.mutation_rate = 0.15
        elif epoch['year'] > 1550:
            kernel.cfg.mutation_rate = 0.08

        # Run epoch
        for _ in range(STEPS_PER_EPOCH):
            kernel.transmit()

        sim_neffs.append(len(kernel.centroids))

    return sim_neffs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("Historical Backtesting (§6.2)")
    print(f"N={N_AGENTS}, {len(HISTORICAL_EPOCHS)} epochs, {N_REPLICATES} replicates")
    print("=" * 60)

    all_trajectories = []
    for rep in range(N_REPLICATES):
        traj = run_single(BASE_SEED + rep)
        all_trajectories.append(traj)
        if (rep + 1) % 5 == 0:
            print(f"  replicate {rep+1}/{N_REPLICATES} done")

    # ── Aggregate ──
    n_epochs = len(HISTORICAL_EPOCHS)
    mean_sim_neffs = []
    std_sim_neffs = []
    for i in range(n_epochs):
        vals = [t[i] for t in all_trajectories]
        mean_sim_neffs.append(statistics.mean(vals))
        std_sim_neffs.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)

    historical_neffs = [e['nEff'] for e in HISTORICAL_EPOCHS]

    # ── Pearson correlation ──
    r, p_value = scipy_stats.pearsonr(historical_neffs, mean_sim_neffs)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Year':>6} {'Label':<28} {'Hist N_eff':>10} {'Sim N_eff':>10} {'±':>6}")
    for i, epoch in enumerate(HISTORICAL_EPOCHS):
        print(f"{epoch['year']:>6} {epoch['label']:<28} {historical_neffs[i]:>10} "
              f"{mean_sim_neffs[i]:>10.1f} {std_sim_neffs[i]:>5.1f}")

    print(f"\nPearson r = {r:.3f}  (p = {p_value:.2e}, n = {n_epochs})")
    print(f"Paper claim: r = 0.82, p < 0.001")

    # ── Save ──
    output = {
        'experiment': 'historical_backtesting',
        'paper_section': '§6.2',
        'paper_claim': 'Pearson r = 0.82 with historical N_eff series (p < 0.001, n = 23)',
        'parameters': {
            'N': N_AGENTS,
            'steps_per_epoch': STEPS_PER_EPOCH,
            'n_replicates': N_REPLICATES,
            'n_epochs': n_epochs,
        },
        'pearson_r': r,
        'p_value': p_value,
        'epochs': [],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    for i, epoch in enumerate(HISTORICAL_EPOCHS):
        output['epochs'].append({
            'year': epoch['year'],
            'label': epoch['label'],
            'coercion': epoch['coercion'],
            'historical_neff': historical_neffs[i],
            'sim_neff_mean': mean_sim_neffs[i],
            'sim_neff_std': std_sim_neffs[i],
        })

    json_path = os.path.join(OUTPUT_DIR, 'backtesting_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        years = [e['year'] for e in HISTORICAL_EPOCHS]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Main plot
        ax1.plot(years, historical_neffs, 'o-', color='#E91E63', linewidth=2,
                 markersize=6, label='Historical estimate', zorder=3)
        ax1.errorbar(years, mean_sim_neffs, yerr=std_sim_neffs, fmt='s-',
                     color='#2196F3', linewidth=2, markersize=5, capsize=3,
                     label=f'Simulation (r={r:.2f})', zorder=2)
        ax1.fill_between(years,
                         [m - s for m, s in zip(mean_sim_neffs, std_sim_neffs)],
                         [m + s for m, s in zip(mean_sim_neffs, std_sim_neffs)],
                         alpha=0.15, color='#2196F3')

        # Mark key events
        events = {-600: 'Axial Age', 400: 'Theodosian\ndecrees', 600: 'Islam',
                  1550: 'Reformation', 2025: 'Modern'}
        for yr, label in events.items():
            ax1.axvline(x=yr, color='gray', linestyle=':', alpha=0.3)
            ax1.annotate(label, xy=(yr, ax1.get_ylim()[1] * 0.95),
                        fontsize=7, ha='center', color='gray')

        ax1.set_ylabel('Effective Deity Count (N_eff)', fontsize=12)
        ax1.set_title(f'Historical Backtesting: Simulation vs History (r={r:.2f}, p={p_value:.1e})',
                      fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Coercion schedule
        coercions = [e['coercion'] for e in HISTORICAL_EPOCHS]
        ax2.fill_between(years, coercions, alpha=0.4, color='#FF9800')
        ax2.plot(years, coercions, '-', color='#FF9800', linewidth=2)
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Coercion (γ)', fontsize=12)
        ax2.set_title('Coercion Schedule', fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, 'backtesting_plot.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved: {plot_path}")
    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == '__main__':
    main()
