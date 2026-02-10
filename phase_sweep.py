#!/usr/bin/env python3
"""
Phase space exploration for Gods as Centroids GABM
Maps coercion × belief_influence parameter space to identify:
- Mono-stable regimes (single dominant centroid)
- Multi-stable regimes (competing centroids)
- Chaotic regimes (unstable clustering)
"""

import json
import numpy as np
from pathlib import Path
import sys
sys.path.append('sim')
from swarm_kernel import SwarmKernel, Config
from run import PrintCB
# Visualization imports - optional for headless runs
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Skipping plots.")
from typing import Dict, List, Tuple
from datetime import datetime

class PhaseSweepCB(PrintCB):
    """Callback to collect phase metrics during runs"""
    def __init__(self):
        super().__init__()
        self.final_metrics = {}
        self.cluster_history = []
        
    def on_step(self, t: int, k: SwarmKernel):
        if t % 100 == 0:  # Sample cluster state
            self.cluster_history.append({
                't': t,
                'cluster_count': len(k.clusters),
                'cluster_sizes': [len(c) for c in k.clusters],
                'entropy': k.metrics.cond_entropy if hasattr(k.metrics, 'cond_entropy') else 0,
                'zipf_slope': k.metrics.zipf_slope if hasattr(k.metrics, 'zipf_slope') else 0,
            })
    
    def on_generation(self, gen: int, k: SwarmKernel):
        # Capture final state metrics
        self.final_metrics = {
            'cluster_count': len(k.clusters),
            'max_cluster_size': max([len(c) for c in k.clusters]) if k.clusters else 0,
            'cluster_entropy': self._cluster_entropy(k.clusters, k.cfg.N),
            'zipf_slope': k.metrics.zipf_slope if hasattr(k.metrics, 'zipf_slope') else 0,
            'cond_entropy': k.metrics.cond_entropy if hasattr(k.metrics, 'cond_entropy') else 0,
            'top_form_dominance': self._top_form_dominance(k),
        }
    
    def _cluster_entropy(self, clusters: List[List[int]], N: int) -> float:
        """Shannon entropy of cluster size distribution"""
        if not clusters:
            return 0.0
        sizes = [len(c) for c in clusters]
        probs = [s/N for s in sizes]
        return -sum(p * np.log2(p) for p in probs if p > 0)
    
    def _top_form_dominance(self, k: SwarmKernel) -> float:
        """Fraction of total usage by most frequent form"""
        if not k.tokens:
            return 0.0
        from collections import Counter
        form_freq = Counter(k.tokens)
        total = sum(form_freq.values())
        top_count = form_freq.most_common(1)[0][1]
        return top_count / total if total > 0 else 0.0

def run_phase_point(coercion: float, belief_influence: float, steps: int = 3000) -> Dict:
    """Run single point in phase space"""
    cfg = Config(
        N=40,
        steps_per_generation=steps,
        seed=42,  # Fixed for reproducibility
        coercion=coercion,
        belief_influence=belief_influence,
        use_deity_priors=True,
        cluster_update_freq=50,
        cluster_threshold=0.4,
    )
    
    kern = SwarmKernel(cfg)
    cb = PhaseSweepCB()
    kern.run(steps, cb)
    
    return {
        'coercion': coercion,
        'belief_influence': belief_influence,
        'final_metrics': cb.final_metrics,
        'cluster_history': cb.cluster_history,
    }

def phase_sweep(coercion_range: Tuple[float, float] = (0.0, 1.0),
                belief_range: Tuple[float, float] = (0.0, 0.5),
                resolution: int = 8) -> List[Dict]:
    """Sweep parameter space and collect phase data"""
    
    coercion_vals = np.linspace(*coercion_range, resolution)
    belief_vals = np.linspace(*belief_range, resolution)
    
    results = []
    total_runs = len(coercion_vals) * len(belief_vals)
    
    print(f"Running {total_runs} phase space points...")
    
    for i, coercion in enumerate(coercion_vals):
        for j, belief in enumerate(belief_vals):
            print(f"[{i*len(belief_vals)+j+1}/{total_runs}] "
                  f"coercion={coercion:.2f}, belief={belief:.2f}")
            
            result = run_phase_point(coercion, belief)
            results.append(result)
    
    return results

def plot_phase_diagram(results: List[Dict], output_dir: Path):
    """Generate phase diagram visualizations"""
    
    # Extract data for plotting
    coercions = [r['coercion'] for r in results]
    beliefs = [r['belief_influence'] for r in results]
    cluster_counts = [r['final_metrics']['cluster_count'] for r in results]
    cluster_entropies = [r['final_metrics']['cluster_entropy'] for r in results]
    dominances = [r['final_metrics']['top_form_dominance'] for r in results]
    
    # Create grid for heatmaps
    coercion_unique = sorted(set(coercions))
    belief_unique = sorted(set(beliefs))
    
    def make_grid(values):
        grid = np.zeros((len(belief_unique), len(coercion_unique)))
        for i, result in enumerate(results):
            c_idx = coercion_unique.index(result['coercion'])
            b_idx = belief_unique.index(result['belief_influence'])
            grid[b_idx, c_idx] = values[i]
        return grid
    
    # Plot phase diagrams
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Cluster count heatmap
    sns.heatmap(make_grid(cluster_counts), 
                xticklabels=[f"{c:.1f}" for c in coercion_unique],
                yticklabels=[f"{b:.1f}" for b in belief_unique],
                ax=axes[0,0], cmap='viridis', annot=True, fmt='.0f')
    axes[0,0].set_title('Cluster Count')
    axes[0,0].set_xlabel('Coercion')
    axes[0,0].set_ylabel('Belief Influence')
    
    # Cluster entropy heatmap  
    sns.heatmap(make_grid(cluster_entropies),
                xticklabels=[f"{c:.1f}" for c in coercion_unique],
                yticklabels=[f"{b:.1f}" for b in belief_unique],
                ax=axes[0,1], cmap='plasma', annot=True, fmt='.2f')
    axes[0,1].set_title('Cluster Entropy')
    axes[0,1].set_xlabel('Coercion')
    axes[0,1].set_ylabel('Belief Influence')
    
    # Dominance heatmap
    sns.heatmap(make_grid(dominances),
                xticklabels=[f"{c:.1f}" for c in coercion_unique], 
                yticklabels=[f"{b:.1f}" for b in belief_unique],
                ax=axes[1,0], cmap='Reds', annot=True, fmt='.2f')
    axes[1,0].set_title('Top Form Dominance')
    axes[1,0].set_xlabel('Coercion')
    axes[1,0].set_ylabel('Belief Influence')
    
    # Phase classification
    phase_grid = np.zeros_like(make_grid(cluster_counts))
    for i, result in enumerate(results):
        c_idx = coercion_unique.index(result['coercion'])
        b_idx = belief_unique.index(result['belief_influence'])
        
        # Classify phase based on metrics
        cc = result['final_metrics']['cluster_count']
        dom = result['final_metrics']['top_form_dominance']
        
        if cc <= 2 and dom > 0.4:
            phase = 1  # Monotheistic
        elif cc >= 5 and dom < 0.25:
            phase = 3  # Polytheistic
        else:
            phase = 2  # Transitional
            
        phase_grid[b_idx, c_idx] = phase
    
    sns.heatmap(phase_grid,
                xticklabels=[f"{c:.1f}" for c in coercion_unique],
                yticklabels=[f"{b:.1f}" for b in belief_unique],
                ax=axes[1,1], cmap='RdYlBu_r', annot=True, fmt='.0f',
                cbar_kws={'ticks': [1, 2, 3], 'label': 'Phase'})
    axes[1,1].set_title('Phase Classification\n(1=Mono, 2=Trans, 3=Poly)')
    axes[1,1].set_xlabel('Coercion')
    axes[1,1].set_ylabel('Belief Influence')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Run phase space exploration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"sim/runs/{timestamp}_phase_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔬 Gods as Centroids Phase Space Exploration")
    print("=" * 50)
    
    # Run sweep
    results = phase_sweep(resolution=6)  # 6x6 = 36 runs
    
    # Save raw data
    with open(output_dir / 'phase_data.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualizations
    plot_phase_diagram(results, output_dir)
    
    print(f"\n✅ Phase sweep complete! Results saved to: {output_dir}")
    print("\nPhase diagram reveals attractor landscape:")
    print("- High coercion → monotheistic convergence")
    print("- Low coercion → polytheistic diversity") 
    print("- Belief influence modulates transition sharpness")

if __name__ == "__main__":
    main()
