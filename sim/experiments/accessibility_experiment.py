#!/usr/bin/env python3
"""
Accessibility Corollary Experiment
Tests the formal prediction that sensory-restricted agents converge to the same 
godform centroids as fully-abled agents, validating channel-invariant attractors.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from swarm_kernel import SwarmKernel, Config
from typing import Dict, List, Tuple
import pandas as pd

def measure_godform_convergence(kernel: SwarmKernel) -> Dict[str, float]:
    """Measure convergence metrics for godform centroids"""
    if not kernel.centroids:
        return {"centroid_count": 0, "max_cluster_size": 0, "convergence_ratio": 0.0}
    
    # Count agents in largest cluster (dominant godform)
    cluster_sizes = [len(cluster) for cluster in kernel.clusters if cluster]
    max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
    convergence_ratio = max_cluster_size / len(kernel.agents) if kernel.agents else 0.0
    
    # Measure centroid stability (distance between centroids)
    centroid_distances = []
    for i, c1 in enumerate(kernel.centroids):
        for j, c2 in enumerate(kernel.centroids[i+1:], i+1):
            from swarm_kernel import cosine
            dist = 1.0 - cosine(c1, c2)
            centroid_distances.append(dist)
    
    avg_centroid_distance = np.mean(centroid_distances) if centroid_distances else 0.0
    
    return {
        "centroid_count": len(kernel.centroids),
        "max_cluster_size": max_cluster_size,
        "convergence_ratio": convergence_ratio,
        "avg_centroid_distance": avg_centroid_distance,
        "cluster_sizes": cluster_sizes
    }

def analyze_sensory_groups(kernel: SwarmKernel) -> Dict[str, Dict]:
    """Analyze convergence by sensory capability groups"""
    from swarm_kernel import SENSORY_CHANNELS
    
    # Group agents by sensory capabilities
    groups = {
        "full": [],      # All channels
        "deafblind": [], # Touch, proprioception, smell, taste
        "deaf": [],      # No sound
        "blind": [],     # No sight
        "anosmia": []    # No smell
    }
    
    for agent in kernel.agents:
        if len(agent.sensory_channels) == len(SENSORY_CHANNELS):
            groups["full"].append(agent)
        elif "sight" not in agent.sensory_channels and "sound" not in agent.sensory_channels:
            groups["deafblind"].append(agent)
        elif "sound" not in agent.sensory_channels:
            groups["deaf"].append(agent)
        elif "sight" not in agent.sensory_channels:
            groups["blind"].append(agent)
        elif "smell" not in agent.sensory_channels:
            groups["anosmia"].append(agent)
    
    # Measure belief convergence within each group
    group_stats = {}
    for group_name, agents in groups.items():
        if not agents:
            continue
            
        # Calculate pairwise belief similarities within group
        similarities = []
        for i, a1 in enumerate(agents):
            for a2 in agents[i+1:]:
                from swarm_kernel import cosine
                sim = cosine(a1.belief, a2.belief)
                similarities.append(sim)
        
        group_stats[group_name] = {
            "count": len(agents),
            "avg_belief_similarity": np.mean(similarities) if similarities else 0.0,
            "belief_coherence": np.std(similarities) if similarities else 0.0
        }
    
    return group_stats

def run_accessibility_experiment():
    """Run the Accessibility Corollary experiment"""
    
    # Load configuration
    with open('sim/configs/accessibility_test.json', 'r') as f:
        config_data = json.load(f)
    
    config = Config(**config_data)
    
    print("🧠 Accessibility Corollary Experiment")
    print(f"Testing {config.N} agents with {config.sensory_restriction_ratio:.1%} sensory restrictions")
    print(f"Coercion: {config.coercion}, Belief influence: {config.belief_influence}")
    print()
    
    # Run simulation
    kernel = SwarmKernel(config)
    
    # Track metrics over time
    time_series = {
        "step": [],
        "centroid_count": [],
        "convergence_ratio": [],
        "full_agents_similarity": [],
        "restricted_agents_similarity": []
    }
    
    steps = 5000
    for step in range(steps):
        kernel.transmit()
        
        if step % 200 == 0:
            convergence = measure_godform_convergence(kernel)
            groups = analyze_sensory_groups(kernel)
            
            time_series["step"].append(step)
            time_series["centroid_count"].append(convergence["centroid_count"])
            time_series["convergence_ratio"].append(convergence["convergence_ratio"])
            time_series["full_agents_similarity"].append(
                groups.get("full", {}).get("avg_belief_similarity", 0.0)
            )
            
            # Average similarity across all restricted groups
            restricted_sims = [
                groups.get(g, {}).get("avg_belief_similarity", 0.0) 
                for g in ["deafblind", "deaf", "blind", "anosmia"]
                if g in groups
            ]
            time_series["restricted_agents_similarity"].append(
                np.mean(restricted_sims) if restricted_sims else 0.0
            )
            
            if step % 1000 == 0:
                print(f"Step {step:4d}: {convergence['centroid_count']} centroids, "
                      f"{convergence['convergence_ratio']:.2%} convergence")
    
    # Final analysis
    print("\n📊 Final Results:")
    final_convergence = measure_godform_convergence(kernel)
    final_groups = analyze_sensory_groups(kernel)
    
    print(f"Godform centroids: {final_convergence['centroid_count']}")
    print(f"Dominant cluster: {final_convergence['max_cluster_size']}/{config.N} agents "
          f"({final_convergence['convergence_ratio']:.1%})")
    print()
    
    print("Sensory Group Analysis:")
    for group, stats in final_groups.items():
        print(f"  {group:10s}: {stats['count']:2d} agents, "
              f"belief similarity = {stats['avg_belief_similarity']:.3f}")
    
    # Test Accessibility Corollary predictions
    print("\n🔬 Accessibility Corollary Test:")
    
    full_sim = final_groups.get("full", {}).get("avg_belief_similarity", 0.0)
    restricted_sims = [
        final_groups.get(g, {}).get("avg_belief_similarity", 0.0) 
        for g in ["deafblind", "deaf", "blind", "anosmia"]
        if g in final_groups and final_groups[g]["count"] > 1
    ]
    
    if restricted_sims:
        avg_restricted_sim = np.mean(restricted_sims)
        similarity_difference = abs(full_sim - avg_restricted_sim)
        
        print(f"Full sensory agents similarity: {full_sim:.3f}")
        print(f"Restricted agents similarity:   {avg_restricted_sim:.3f}")
        print(f"Difference: {similarity_difference:.3f}")
        
        # Accessibility Corollary prediction: difference should be small
        threshold = 0.1  # Empirical threshold for "channel-invariant"
        if similarity_difference < threshold:
            print("✅ ACCESSIBILITY COROLLARY SUPPORTED")
            print("   Godform convergence is channel-invariant")
        else:
            print("❌ ACCESSIBILITY COROLLARY CHALLENGED")
            print("   Significant divergence between sensory groups")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Convergence over time
    plt.subplot(2, 2, 1)
    plt.plot(time_series["step"], time_series["convergence_ratio"], 'b-', linewidth=2)
    plt.xlabel("Simulation Step")
    plt.ylabel("Convergence Ratio")
    plt.title("Godform Convergence Over Time")
    plt.grid(True, alpha=0.3)
    
    # Centroid count over time
    plt.subplot(2, 2, 2)
    plt.plot(time_series["step"], time_series["centroid_count"], 'r-', linewidth=2)
    plt.xlabel("Simulation Step")
    plt.ylabel("Number of Centroids")
    plt.title("Godform Centroids Over Time")
    plt.grid(True, alpha=0.3)
    
    # Belief similarity comparison
    plt.subplot(2, 2, 3)
    plt.plot(time_series["step"], time_series["full_agents_similarity"], 
             'g-', linewidth=2, label="Full sensory")
    plt.plot(time_series["step"], time_series["restricted_agents_similarity"], 
             'orange', linewidth=2, label="Restricted sensory")
    plt.xlabel("Simulation Step")
    plt.ylabel("Belief Similarity")
    plt.title("Accessibility Corollary Test")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final group comparison
    plt.subplot(2, 2, 4)
    groups_to_plot = [(g, s["avg_belief_similarity"]) for g, s in final_groups.items() 
                      if s["count"] > 1]
    if groups_to_plot:
        groups, sims = zip(*groups_to_plot)
        colors = ['blue' if g == 'full' else 'orange' for g in groups]
        plt.bar(groups, sims, color=colors, alpha=0.7)
        plt.ylabel("Belief Similarity")
        plt.title("Final Group Similarities")
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("accessibility_corollary_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return kernel, time_series, final_groups

if __name__ == "__main__":
    kernel, time_series, groups = run_accessibility_experiment()
    
    print("\n📝 Experiment complete. Results saved to accessibility_corollary_results.png")
    print("\nTheoretical Implications:")
    print("- If AC holds: Sensory restrictions don't prevent godform convergence")
    print("- If AC fails: Channel-specific attractors emerge")
    print("- Empirical test: Compare deafblind communities with broader religious patterns")
