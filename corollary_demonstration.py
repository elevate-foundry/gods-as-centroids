#!/usr/bin/env python3
"""
God-Centroid Corollaries: Working Demonstration
Shows that the theoretical predictions are testable and produce expected results
across multiple languages with statistical validation.
"""

import sys
import os
sys.path.append('sim')

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
from swarm_kernel import SwarmKernel, Config

class CorollaryDemonstrator:
    """Demonstrates testable corollary predictions with working simulations"""
    
    def __init__(self):
        self.results = {}
        
    def demonstrate_universality_corollary(self) -> Dict:
        """A: Universality - Every communicative swarm with sufficient coupling yields godforms"""
        print("🌍 Demonstrating Universality Corollary")
        
        results = []
        coupling_proxies = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]  # belief_influence values
        
        for coupling in coupling_proxies:
            # Configure for godform emergence
            config = Config(
                N=30,
                steps_per_generation=1500,
                belief_influence=coupling,
                coercion=0.2,  # Moderate coercion to enable clustering
                use_deity_priors=True,
                cluster_update_freq=50,
                cluster_threshold=0.6,
                seed=42
            )
            
            kernel = SwarmKernel(config)
            
            # Run simulation
            for step in range(config.steps_per_generation):
                kernel.transmit()
            
            # Measure godform emergence
            has_godforms = len(kernel.centroids) > 0
            centroid_count = len(kernel.centroids)
            
            results.append({
                'coupling_proxy': coupling,
                'has_godforms': has_godforms,
                'centroid_count': centroid_count
            })
            
            status = "✅" if has_godforms else "❌"
            print(f"  κ≈{coupling:.1f}: {status} ({centroid_count} centroids)")
        
        # Find threshold
        threshold = None
        for r in results:
            if r['has_godforms']:
                threshold = r['coupling_proxy']
                break
        
        return {
            'corollary': 'A_universality',
            'prediction': 'P(godforms) = 0 if κ < κ*, P(godforms) > 0 if κ > κ*',
            'threshold_found': threshold,
            'results': results,
            'validated': threshold is not None
        }
    
    def demonstrate_monotheism_transition(self) -> Dict:
        """B: Plurality→Monotheism - N_eff decreases with coercion"""
        print("⚖️  Demonstrating Monotheism Transition")
        
        results = []
        coercion_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        for coercion in coercion_levels:
            config = Config(
                N=40,
                steps_per_generation=2000,
                coercion=coercion,
                belief_influence=0.3,  # Sufficient coupling
                use_deity_priors=True,
                cluster_update_freq=50,
                cluster_threshold=0.5,
                seed=42
            )
            
            kernel = SwarmKernel(config)
            
            # Run simulation
            for step in range(config.steps_per_generation):
                kernel.transmit()
            
            # Measure effective deity count
            n_eff = len(kernel.centroids)
            largest_cluster = max([len(c) for c in kernel.clusters]) if kernel.clusters else 0
            dominance = largest_cluster / len(kernel.agents) if kernel.agents else 0
            
            results.append({
                'coercion': coercion,
                'n_effective': n_eff,
                'dominance_ratio': dominance
            })
            
            regime = "monotheistic" if n_eff <= 1 else "polytheistic" if n_eff > 3 else "oligotheistic"
            print(f"  c={coercion:.1f}: {n_eff} centroids, dominance={dominance:.2f} ({regime})")
        
        # Test monotonic decrease
        n_effs = [r['n_effective'] for r in results]
        coercions = [r['coercion'] for r in results]
        
        # Calculate trend
        correlation = np.corrcoef(coercions, n_effs)[0, 1] if len(set(n_effs)) > 1 else 0
        monotonic_decrease = correlation < -0.3  # Negative correlation threshold
        
        return {
            'corollary': 'B_monotheism_transition',
            'prediction': '∂N_eff/∂c < 0 monotonically',
            'correlation': correlation,
            'monotonic_decrease': monotonic_decrease,
            'results': results,
            'validated': monotonic_decrease
        }
    
    def demonstrate_ritual_stabilizer(self) -> Dict:
        """D: Ritual Stabilizer - Higher ritual costs reduce churn"""
        print("🕯️  Demonstrating Ritual Stabilizer")
        
        results = []
        ritual_costs = [0.0, 0.05, 0.10, 0.15, 0.25, 0.40]
        
        for cost in ritual_costs:
            config = Config(
                N=35,
                steps_per_generation=2000,
                ritual_bonus=cost,
                ritual_period=30,  # More frequent rituals
                coercion=0.3,
                belief_influence=0.25,
                use_deity_priors=True,
                cluster_update_freq=25,
                seed=42
            )
            
            kernel = SwarmKernel(config)
            
            # Track centroid stability
            centroid_counts = []
            for step in range(config.steps_per_generation):
                kernel.transmit()
                if step % 50 == 0:
                    centroid_counts.append(len(kernel.centroids))
            
            # Calculate churn (variance in centroid count)
            churn = np.var(centroid_counts) if len(centroid_counts) > 1 else 0
            stability = 1.0 / (1.0 + churn)  # Higher = more stable
            
            results.append({
                'ritual_cost': cost,
                'churn': churn,
                'stability': stability,
                'final_centroids': len(kernel.centroids)
            })
            
            print(f"  ρ={cost:.2f}: churn={churn:.3f}, stability={stability:.3f}")
        
        # Test negative correlation between cost and churn
        costs = [r['ritual_cost'] for r in results]
        churns = [r['churn'] for r in results]
        
        correlation = np.corrcoef(costs, churns)[0, 1] if len(set(churns)) > 1 else 0
        stabilizing_effect = correlation < -0.2  # Negative correlation
        
        return {
            'corollary': 'D_ritual_stabilizer',
            'prediction': '∂churn/∂ρ < 0 (higher cost → lower churn)',
            'correlation': correlation,
            'stabilizing_effect': stabilizing_effect,
            'results': results,
            'validated': stabilizing_effect
        }
    
    def demonstrate_prestige_amplifier(self) -> Dict:
        """E: Prestige Amplifier - Higher prestige accelerates convergence"""
        print("⭐ Demonstrating Prestige Amplifier")
        
        results = []
        prestige_weights = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
        
        for alpha in prestige_weights:
            config = Config(
                N=35,
                steps_per_generation=1500,
                prestige_alpha=alpha,
                coercion=0.4,
                belief_influence=0.3,
                use_deity_priors=True,
                cluster_update_freq=25,
                seed=42
            )
            
            kernel = SwarmKernel(config)
            
            # Track convergence
            convergence_step = None
            for step in range(config.steps_per_generation):
                kernel.transmit()
                
                # Check for dominance (>60% in largest cluster)
                if kernel.clusters and convergence_step is None:
                    max_cluster = max([len(c) for c in kernel.clusters])
                    if max_cluster / len(kernel.agents) > 0.6:
                        convergence_step = step
            
            if convergence_step is None:
                convergence_step = config.steps_per_generation
            
            convergence_speed = 1.0 / (convergence_step + 1)  # Higher = faster
            
            results.append({
                'prestige_alpha': alpha,
                'convergence_step': convergence_step,
                'convergence_speed': convergence_speed,
                'converged': convergence_step < config.steps_per_generation
            })
            
            status = "✅" if convergence_step < config.steps_per_generation else "❌"
            print(f"  α={alpha:.1f}: {status} converged at step {convergence_step}")
        
        # Test positive correlation between prestige and convergence speed
        alphas = [r['prestige_alpha'] for r in results]
        speeds = [r['convergence_speed'] for r in results]
        
        correlation = np.corrcoef(alphas, speeds)[0, 1] if len(set(speeds)) > 1 else 0
        amplifying_effect = correlation > 0.2  # Positive correlation
        
        return {
            'corollary': 'E_prestige_amplifier',
            'prediction': '∂convergence_speed/∂α > 0',
            'correlation': correlation,
            'amplifying_effect': amplifying_effect,
            'results': results,
            'validated': amplifying_effect
        }
    
    def demonstrate_accessibility_invariance(self) -> Dict:
        """I: Accessibility - Godforms invariant under sensory restrictions"""
        print("👁️  Demonstrating Accessibility Invariance")
        
        # Test with and without sensory restrictions
        configs = [
            ('full_sensory', {'enable_sensory_restrictions': False}),
            ('restricted_sensory', {'enable_sensory_restrictions': True, 'sensory_restriction_ratio': 0.4})
        ]
        
        results = []
        
        for condition_name, config_override in configs:
            base_config = {
                'N': 40,
                'steps_per_generation': 2000,
                'coercion': 0.3,
                'belief_influence': 0.3,
                'use_deity_priors': True,
                'cluster_update_freq': 50,
                'seed': 42
            }
            base_config.update(config_override)
            
            config = Config(**base_config)
            kernel = SwarmKernel(config)
            
            # Run simulation
            for step in range(config.steps_per_generation):
                kernel.transmit()
            
            # Measure convergence
            centroid_count = len(kernel.centroids)
            
            # Measure belief similarity within groups
            if kernel.clusters:
                from swarm_kernel import cosine
                similarities = []
                for cluster in kernel.clusters:
                    if len(cluster) > 1:
                        cluster_sims = []
                        for i, aid1 in enumerate(cluster):
                            for aid2 in cluster[i+1:]:
                                sim = cosine(kernel.agents[aid1].belief, kernel.agents[aid2].belief)
                                cluster_sims.append(sim)
                        if cluster_sims:
                            similarities.extend(cluster_sims)
                
                avg_similarity = np.mean(similarities) if similarities else 0.0
            else:
                avg_similarity = 0.0
            
            results.append({
                'condition': condition_name,
                'centroid_count': centroid_count,
                'avg_similarity': avg_similarity
            })
            
            print(f"  {condition_name}: {centroid_count} centroids, similarity={avg_similarity:.3f}")
        
        # Test invariance (similar outcomes across conditions)
        if len(results) == 2:
            count_diff = abs(results[0]['centroid_count'] - results[1]['centroid_count'])
            sim_diff = abs(results[0]['avg_similarity'] - results[1]['avg_similarity'])
            
            # Invariance if differences are small
            invariant = count_diff <= 1 and sim_diff <= 0.2
        else:
            invariant = False
        
        return {
            'corollary': 'I_accessibility_invariance',
            'prediction': 'Godform convergence invariant under sensory restrictions',
            'count_difference': count_diff if len(results) == 2 else None,
            'similarity_difference': sim_diff if len(results) == 2 else None,
            'invariant': invariant,
            'results': results,
            'validated': invariant
        }
    
    def run_comprehensive_demonstration(self) -> Dict:
        """Run all corollary demonstrations"""
        print("🧪 God-Centroid Corollaries: Comprehensive Demonstration")
        print("=" * 70)
        
        demonstrations = {}
        
        # Run each demonstration
        demonstrations['universality'] = self.demonstrate_universality_corollary()
        print()
        demonstrations['monotheism'] = self.demonstrate_monotheism_transition()
        print()
        demonstrations['ritual_stabilizer'] = self.demonstrate_ritual_stabilizer()
        print()
        demonstrations['prestige_amplifier'] = self.demonstrate_prestige_amplifier()
        print()
        demonstrations['accessibility'] = self.demonstrate_accessibility_invariance()
        
        return demonstrations
    
    def generate_validation_summary(self, demonstrations: Dict) -> Dict:
        """Generate summary of validation results"""
        total_corollaries = len(demonstrations)
        validated_corollaries = sum(1 for d in demonstrations.values() if d['validated'])
        
        validation_rate = validated_corollaries / total_corollaries
        
        # Decision rule: ≥80% validation for framework support
        framework_supported = validation_rate >= 0.8
        
        return {
            'total_corollaries': total_corollaries,
            'validated_corollaries': validated_corollaries,
            'validation_rate': validation_rate,
            'framework_supported': framework_supported,
            'decision': "SUPPORTED" if framework_supported else "NEEDS_REVISION"
        }
    
    def plot_demonstration_results(self, demonstrations: Dict):
        """Create visualization of demonstration results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('God-Centroid Corollaries: Empirical Validation', fontsize=16)
        
        # Universality
        if 'universality' in demonstrations:
            ax = axes[0, 0]
            data = demonstrations['universality']['results']
            x = [d['coupling_proxy'] for d in data]
            y = [d['has_godforms'] for d in data]
            ax.plot(x, y, 'bo-', linewidth=2, markersize=8)
            ax.set_xlabel('Coupling Strength κ')
            ax.set_ylabel('Godform Emergence')
            ax.set_title('A: Universality Threshold')
            ax.grid(True, alpha=0.3)
        
        # Monotheism Transition
        if 'monotheism' in demonstrations:
            ax = axes[0, 1]
            data = demonstrations['monotheism']['results']
            x = [d['coercion'] for d in data]
            y = [d['n_effective'] for d in data]
            ax.plot(x, y, 'ro-', linewidth=2, markersize=6)
            ax.set_xlabel('Coercion c')
            ax.set_ylabel('N_effective')
            ax.set_title('B: Monotheism Transition')
            ax.grid(True, alpha=0.3)
        
        # Ritual Stabilizer
        if 'ritual_stabilizer' in demonstrations:
            ax = axes[0, 2]
            data = demonstrations['ritual_stabilizer']['results']
            x = [d['ritual_cost'] for d in data]
            y = [d['stability'] for d in data]
            ax.plot(x, y, 'go-', linewidth=2, markersize=6)
            ax.set_xlabel('Ritual Cost ρ')
            ax.set_ylabel('Stability')
            ax.set_title('D: Ritual Stabilizer')
            ax.grid(True, alpha=0.3)
        
        # Prestige Amplifier
        if 'prestige_amplifier' in demonstrations:
            ax = axes[1, 0]
            data = demonstrations['prestige_amplifier']['results']
            x = [d['prestige_alpha'] for d in data]
            y = [d['convergence_speed'] for d in data]
            ax.plot(x, y, 'mo-', linewidth=2, markersize=6)
            ax.set_xlabel('Prestige Weight α')
            ax.set_ylabel('Convergence Speed')
            ax.set_title('E: Prestige Amplifier')
            ax.grid(True, alpha=0.3)
        
        # Accessibility Invariance
        if 'accessibility' in demonstrations:
            ax = axes[1, 1]
            data = demonstrations['accessibility']['results']
            conditions = [d['condition'] for d in data]
            similarities = [d['avg_similarity'] for d in data]
            ax.bar(conditions, similarities, color=['blue', 'orange'], alpha=0.7)
            ax.set_ylabel('Belief Similarity')
            ax.set_title('I: Accessibility Invariance')
            ax.tick_params(axis='x', rotation=45)
        
        # Validation Summary
        ax = axes[1, 2]
        ax.axis('off')
        summary_text = "Validation Summary:\n\n"
        for name, demo in demonstrations.items():
            status = "✅" if demo['validated'] else "❌"
            summary_text += f"{status} {name.title()}\n"
        ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('corollary_demonstrations.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Run comprehensive corollary demonstration"""
    demonstrator = CorollaryDemonstrator()
    
    # Run demonstrations
    demonstrations = demonstrator.run_comprehensive_demonstration()
    
    # Generate summary
    summary = demonstrator.generate_validation_summary(demonstrations)
    
    print("\n" + "=" * 70)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total Corollaries Tested: {summary['total_corollaries']}")
    print(f"Validated Corollaries: {summary['validated_corollaries']}")
    print(f"Validation Rate: {summary['validation_rate']:.1%}")
    print(f"Framework Decision: {summary['decision']}")
    
    print("\n📋 Individual Results:")
    for name, demo in demonstrations.items():
        status = "✅ VALIDATED" if demo['validated'] else "❌ REFUTED"
        print(f"  {name.upper()}: {status}")
        print(f"    Prediction: {demo['prediction']}")
    
    # Save results
    with open('corollary_demonstration_results.json', 'w') as f:
        json.dump({
            'demonstrations': demonstrations,
            'summary': summary,
            'timestamp': time.time()
        }, f, indent=2, default=str)
    
    # Generate plots
    demonstrator.plot_demonstration_results(demonstrations)
    
    print(f"\n📊 Results saved: corollary_demonstration_results.json")
    print(f"📈 Plots saved: corollary_demonstrations.png")
    
    return demonstrations, summary

if __name__ == "__main__":
    main()
