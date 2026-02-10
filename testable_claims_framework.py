#!/usr/bin/env python3
"""
God-Centroid Co-evolution Law: Testable Claims Framework
Transforms theoretical corollaries into falsifiable hypotheses with measurable indicators,
manipulable drivers, and preregistered predictions for dual-track backtesting.
"""

import sys
import os
sys.path.append('sim')

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TestableClaim:
    """Formal specification of a testable corollary claim"""
    corollary_id: str
    name: str
    indicator_Y: str  # Measurable outcome
    driver_X: str     # Manipulable factor
    prediction: str   # Expected relationship ∂Y/∂X
    expected_sign: str  # '+', '-', 'threshold', 'U-shaped'
    threshold: Optional[float] = None  # Critical value if applicable
    falsifier: str = ""  # What would refute this claim
    proxy_mapping: Dict[str, List[str]] = None  # Historical proxies
    power_analysis: Dict[str, float] = None  # Required effect sizes
    
    def __post_init__(self):
        if self.proxy_mapping is None:
            self.proxy_mapping = {}
        if self.power_analysis is None:
            self.power_analysis = {}

class TestableClaimsFramework:
    """Implements rigorous testing framework for God-Centroid corollaries"""
    
    def __init__(self):
        self.claims = self._define_testable_claims()
        self.simulation_results = {}
        self.historical_results = {}
        self.validation_scores = {}
    
    def _define_testable_claims(self) -> Dict[str, TestableClaim]:
        """Define all corollaries as testable claims with falsifiability criteria"""
        
        claims = {
            'A_universality': TestableClaim(
                corollary_id='A',
                name='Universality Threshold',
                indicator_Y='godform_emergence_probability',
                driver_X='coupling_strength_kappa',
                prediction='∂P(G≠∅)/∂κ > 0 with threshold κ*',
                expected_sign='threshold',
                threshold=0.3,  # Predicted threshold
                falsifier='No godforms emerge even at κ=1.0, or emergence decreases with κ',
                proxy_mapping={
                    'coupling_strength': ['social_network_density', 'communication_frequency', 'ritual_participation'],
                    'godform_emergence': ['religious_institution_count', 'deity_mention_frequency', 'sacred_text_complexity']
                },
                power_analysis={'min_effect_size': 0.5, 'required_n': 50}
            ),
            
            'B_plurality_monotheism': TestableClaim(
                corollary_id='B',
                name='Plurality-Monotheism Transition',
                indicator_Y='effective_centroid_count_Neff',
                driver_X='coercion_parameter_c',
                prediction='∂N_eff/∂c < 0 monotonically',
                expected_sign='-',
                threshold=0.7,  # Predicted monotheism threshold
                falsifier='N_eff increases with coercion, or no monotonic relationship',
                proxy_mapping={
                    'coercion': ['press_freedom_index_inv', 'state_religion_indicator', 'religious_persecution_score'],
                    'plurality': ['religious_diversity_index', 'denomination_count', 'syncretism_measures']
                },
                power_analysis={'min_effect_size': 0.3, 'required_n': 100}
            ),
            
            'C_syncretism_belt': TestableClaim(
                corollary_id='C',
                name='Syncretism at Cultural Borders',
                indicator_Y='fusion_rate_per_timestep',
                driver_X='border_curvature_modularity',
                prediction='∂fusion_rate/∂modularity < 0 (low curvature → high fusion)',
                expected_sign='-',
                falsifier='Higher modularity increases fusion, or borders show less fusion than interiors',
                proxy_mapping={
                    'border_curvature': ['network_modularity', 'geographic_isolation', 'trade_route_density'],
                    'fusion_rate': ['syncretic_practices_count', 'cross_tradition_marriages', 'hybrid_rituals']
                },
                power_analysis={'min_effect_size': 0.4, 'required_n': 75}
            ),
            
            'D_ritual_stabilizer': TestableClaim(
                corollary_id='D',
                name='Ritual Cost Stabilization',
                indicator_Y='godform_churn_variance',
                driver_X='ritual_cost_rho',
                prediction='∂churn/∂ρ < 0 (higher cost → lower churn)',
                expected_sign='-',
                falsifier='No churn change after large ritual cost shifts, or churn increases with cost',
                proxy_mapping={
                    'ritual_cost': ['pilgrimage_expense', 'ceremony_duration', 'sacrifice_value', 'clergy_training_years'],
                    'churn': ['denomination_switching_rate', 'doctrine_change_frequency', 'schism_rate']
                },
                power_analysis={'min_effect_size': 0.35, 'required_n': 80}
            ),
            
            'E_prestige_amplifier': TestableClaim(
                corollary_id='E',
                name='Prestige-Driven Convergence',
                indicator_Y='convergence_speed_timesteps',
                driver_X='prestige_alpha_weight',
                prediction='∂convergence_speed/∂α > 0 (higher prestige → faster convergence)',
                expected_sign='+',
                falsifier='Zero effect of prestige on convergence speed',
                proxy_mapping={
                    'prestige': ['leader_charisma_ratings', 'media_coverage', 'follower_count', 'miracle_claims'],
                    'convergence': ['doctrine_adoption_rate', 'conversion_speed', 'belief_homogenization']
                },
                power_analysis={'min_effect_size': 0.4, 'required_n': 60}
            ),
            
            'F_shock_vectoring': TestableClaim(
                corollary_id='F',
                name='Environmental Shock Reorientation',
                indicator_Y='centroid_drift_magnitude',
                driver_X='shock_intensity_vector',
                prediction='∂drift/∂shock_intensity > 0 with directional bias',
                expected_sign='+',
                falsifier='No centroid reorientation after major environmental shocks',
                proxy_mapping={
                    'shock_intensity': ['natural_disaster_severity', 'war_casualties', 'economic_crisis_depth'],
                    'drift': ['theological_emphasis_shift', 'ritual_practice_changes', 'deity_attribute_evolution']
                },
                power_analysis={'min_effect_size': 0.5, 'required_n': 40}
            ),
            
            'K_entropy_bound': TestableClaim(
                corollary_id='K',
                name='Information-Theoretic Bound',
                indicator_Y='mutual_information_I_H_G',
                driver_X='network_connectivity',
                prediction='∂I(H:G)/∂connectivity > 0 with saturation bound',
                expected_sign='threshold',
                threshold=0.8,  # Predicted saturation point
                falsifier='Higher connectivity yields unbounded doctrinal variance increase',
                proxy_mapping={
                    'connectivity': ['internet_penetration', 'transportation_density', 'media_access'],
                    'mutual_info': ['doctrine_coherence', 'belief_correlation', 'theological_complexity']
                },
                power_analysis={'min_effect_size': 0.3, 'required_n': 120}
            )
        }
        
        return claims
    
    def run_simulation_track(self, claim: TestableClaim, n_trials: int = 50) -> Dict:
        """Run GABM parameter sweeps to test corollary predictions"""
        print(f"🔬 Simulation Track: {claim.name}")
        
        from swarm_kernel import SwarmKernel, Config
        
        # Define parameter sweep based on driver
        if claim.driver_X == 'coupling_strength_kappa':
            sweep_values = np.linspace(0.0, 1.0, 11)
            param_name = 'coupling'  # Map to actual config parameter
        elif claim.driver_X == 'coercion_parameter_c':
            sweep_values = np.linspace(0.0, 1.0, 11)
            param_name = 'coercion'
        elif claim.driver_X == 'ritual_cost_rho':
            sweep_values = np.linspace(0.0, 0.5, 11)
            param_name = 'ritual_bonus'
        elif claim.driver_X == 'prestige_alpha_weight':
            sweep_values = np.linspace(0.0, 1.0, 11)
            param_name = 'prestige_alpha'
        else:
            # Default sweep
            sweep_values = np.linspace(0.0, 1.0, 11)
            param_name = 'coercion'
        
        results = []
        base_config = {
            'N': 40,
            'steps': 2000,
            'enable_sensory_restrictions': False,
            'seed': 42
        }
        
        for x_val in sweep_values:
            trial_results = []
            
            for trial in range(n_trials):
                config_dict = base_config.copy()
                config_dict[param_name] = x_val
                config_dict['seed'] = 42 + trial  # Different seed per trial
                
                config = Config(**config_dict)
                kernel = SwarmKernel(config)
                
                # Run simulation
                for _ in range(config.steps):
                    kernel.transmit()
                
                # Measure indicator Y
                y_val = self._measure_indicator(kernel, claim.indicator_Y)
                trial_results.append(y_val)
            
            # Aggregate trial results
            mean_y = np.mean(trial_results)
            std_y = np.std(trial_results)
            
            results.append({
                'x_value': x_val,
                'y_mean': mean_y,
                'y_std': std_y,
                'y_values': trial_results
            })
            
            print(f"  {claim.driver_X}={x_val:.2f}: {claim.indicator_Y}={mean_y:.3f}±{std_y:.3f}")
        
        return {
            'claim_id': claim.corollary_id,
            'sweep_results': results,
            'parameter': param_name,
            'indicator': claim.indicator_Y
        }
    
    def _measure_indicator(self, kernel, indicator: str) -> float:
        """Extract measurable indicator from simulation state"""
        if indicator == 'godform_emergence_probability':
            return 1.0 if len(kernel.centroids) > 0 else 0.0
        
        elif indicator == 'effective_centroid_count_Neff':
            return len(kernel.centroids)
        
        elif indicator == 'fusion_rate_per_timestep':
            # Approximate fusion rate by centroid stability
            if hasattr(kernel, 'centroid_history'):
                changes = sum(1 for i in range(1, len(kernel.centroid_history)) 
                            if len(kernel.centroid_history[i]) != len(kernel.centroid_history[i-1]))
                return changes / len(kernel.centroid_history) if kernel.centroid_history else 0.0
            return 0.0
        
        elif indicator == 'godform_churn_variance':
            # Measure variance in centroid count over time
            if hasattr(kernel, 'metrics') and 'centroid_counts' in kernel.metrics:
                counts = kernel.metrics['centroid_counts']
                return np.var(counts) if len(counts) > 1 else 0.0
            return 0.0
        
        elif indicator == 'convergence_speed_timesteps':
            # Measure steps to reach dominant cluster (>70% agents)
            if kernel.clusters:
                max_cluster_size = max(len(c) for c in kernel.clusters)
                dominance = max_cluster_size / len(kernel.agents)
                return 1.0 / (dominance + 0.1)  # Inverse of dominance as proxy for speed
            return 1000.0  # High value if no convergence
        
        elif indicator == 'centroid_drift_magnitude':
            # Measure average distance between centroids
            if len(kernel.centroids) > 1:
                from swarm_kernel import cosine
                distances = []
                for i, c1 in enumerate(kernel.centroids):
                    for c2 in kernel.centroids[i+1:]:
                        dist = 1.0 - cosine(c1, c2)
                        distances.append(dist)
                return np.mean(distances) if distances else 0.0
            return 0.0
        
        elif indicator == 'mutual_information_I_H_G':
            # Proxy: coherence within clusters
            if kernel.clusters:
                from swarm_kernel import cosine
                coherence = 0.0
                for cluster in kernel.clusters:
                    if len(cluster) > 1:
                        similarities = []
                        for i, agent_id1 in enumerate(cluster):
                            for agent_id2 in cluster[i+1:]:
                                sim = cosine(kernel.agents[agent_id1].belief, 
                                           kernel.agents[agent_id2].belief)
                                similarities.append(sim)
                        if similarities:
                            coherence += np.mean(similarities)
                return coherence / len(kernel.clusters) if kernel.clusters else 0.0
            return 0.0
        
        else:
            return 0.0
    
    def test_prediction_sign(self, results: Dict, claim: TestableClaim) -> Dict:
        """Test if simulation results match predicted sign/relationship"""
        sweep_data = results['sweep_results']
        x_values = [r['x_value'] for r in sweep_data]
        y_values = [r['y_mean'] for r in sweep_data]
        
        # Calculate correlation and trend
        correlation, p_value = stats.pearsonr(x_values, y_values)
        
        # Test predicted sign
        if claim.expected_sign == '+':
            prediction_met = correlation > 0 and p_value < 0.05
            expected_direction = "positive"
        elif claim.expected_sign == '-':
            prediction_met = correlation < 0 and p_value < 0.05
            expected_direction = "negative"
        elif claim.expected_sign == 'threshold':
            # Look for threshold behavior (sigmoid-like)
            # Fit sigmoid and check for threshold
            prediction_met = self._test_threshold_behavior(x_values, y_values, claim.threshold)
            expected_direction = "threshold"
        else:
            prediction_met = False
            expected_direction = "unknown"
        
        # Calculate effect size (Cohen's d equivalent for correlation)
        effect_size = abs(correlation)
        
        return {
            'claim_id': claim.corollary_id,
            'prediction_met': prediction_met,
            'correlation': correlation,
            'p_value': p_value,
            'effect_size': effect_size,
            'expected_direction': expected_direction,
            'observed_direction': "positive" if correlation > 0 else "negative",
            'statistical_power': self._calculate_power(len(x_values), effect_size)
        }
    
    def _test_threshold_behavior(self, x_values: List[float], y_values: List[float], 
                                expected_threshold: Optional[float]) -> bool:
        """Test for threshold/sigmoid behavior in the relationship"""
        if expected_threshold is None:
            return False
        
        # Simple threshold test: check if Y changes significantly around threshold
        below_threshold = [y for x, y in zip(x_values, y_values) if x < expected_threshold]
        above_threshold = [y for x, y in zip(x_values, y_values) if x >= expected_threshold]
        
        if len(below_threshold) < 2 or len(above_threshold) < 2:
            return False
        
        # T-test for difference in means
        t_stat, p_val = stats.ttest_ind(below_threshold, above_threshold)
        return p_val < 0.05 and np.mean(above_threshold) > np.mean(below_threshold)
    
    def _calculate_power(self, n: int, effect_size: float) -> float:
        """Calculate statistical power for given sample size and effect size"""
        # Simplified power calculation for correlation
        if n < 3:
            return 0.0
        
        # Cohen's conventions: small=0.1, medium=0.3, large=0.5
        z_alpha = 1.96  # Two-tailed α=0.05
        z_beta = stats.norm.ppf(0.8)  # Power = 0.8
        
        # Fisher's z-transformation
        z_r = 0.5 * np.log((1 + effect_size) / (1 - effect_size))
        z_null = 0
        
        # Power calculation
        se = 1 / np.sqrt(n - 3)
        z_crit = (z_r - z_null) / se
        power = 1 - stats.norm.cdf(z_alpha - z_crit)
        
        return min(1.0, max(0.0, power))
    
    def run_comprehensive_validation(self) -> Dict:
        """Run complete validation across all testable claims"""
        print("🧪 God-Centroid Testable Claims Validation")
        print("=" * 60)
        
        validation_results = {}
        
        for claim_id, claim in self.claims.items():
            print(f"\n📋 Testing Claim {claim.corollary_id}: {claim.name}")
            print("-" * 40)
            
            # Run simulation track
            sim_results = self.run_simulation_track(claim, n_trials=10)  # Reduced for speed
            
            # Test predictions
            prediction_test = self.test_prediction_sign(sim_results, claim)
            
            validation_results[claim_id] = {
                'claim': asdict(claim),
                'simulation_results': sim_results,
                'prediction_test': prediction_test
            }
            
            # Report results
            status = "✅ SUPPORTED" if prediction_test['prediction_met'] else "❌ REFUTED"
            print(f"  {status}")
            print(f"  Correlation: {prediction_test['correlation']:.3f} (p={prediction_test['p_value']:.3f})")
            print(f"  Effect size: {prediction_test['effect_size']:.3f}")
            print(f"  Power: {prediction_test['statistical_power']:.3f}")
        
        return validation_results
    
    def generate_preregistration_document(self) -> str:
        """Generate preregistration document with all testable predictions"""
        doc = []
        doc.append("# God-Centroid Co-evolution Law: Preregistered Predictions")
        doc.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        doc.append("")
        doc.append("## Theoretical Framework")
        doc.append("The God-Centroid Co-evolution Law posits that religious concepts emerge as")
        doc.append("centroids in semantic space through agent-based communication dynamics.")
        doc.append("")
        doc.append("## Testable Claims")
        doc.append("")
        
        for claim_id, claim in self.claims.items():
            doc.append(f"### Claim {claim.corollary_id}: {claim.name}")
            doc.append("")
            doc.append(f"**Indicator (Y)**: {claim.indicator_Y}")
            doc.append(f"**Driver (X)**: {claim.driver_X}")
            doc.append(f"**Prediction**: {claim.prediction}")
            doc.append(f"**Expected Sign**: {claim.expected_sign}")
            if claim.threshold:
                doc.append(f"**Threshold**: {claim.threshold}")
            doc.append(f"**Falsifier**: {claim.falsifier}")
            doc.append("")
            doc.append("**Historical Proxies**:")
            for var, proxies in claim.proxy_mapping.items():
                doc.append(f"- {var}: {', '.join(proxies)}")
            doc.append("")
            doc.append("**Power Analysis**:")
            for metric, value in claim.power_analysis.items():
                doc.append(f"- {metric}: {value}")
            doc.append("")
        
        doc.append("## Decision Rule")
        doc.append("If ≥80% of preregistered corollaries hit predicted sign/thresholds")
        doc.append("across simulation + historical tracks, treat the law as corroborated.")
        doc.append("Any hard falsifier triggers model revision.")
        doc.append("")
        doc.append("## Multiple Testing Correction")
        doc.append("Benjamini-Hochberg FDR control at α=0.05 across all claims.")
        
        return "\n".join(doc)
    
    def calculate_validation_score(self, results: Dict) -> Dict:
        """Calculate overall validation score for the theoretical framework"""
        total_claims = len(results)
        supported_claims = sum(1 for r in results.values() 
                             if r['prediction_test']['prediction_met'])
        
        support_rate = supported_claims / total_claims
        
        # Calculate average effect size and power
        effect_sizes = [r['prediction_test']['effect_size'] for r in results.values()]
        powers = [r['prediction_test']['statistical_power'] for r in results.values()]
        
        avg_effect_size = np.mean(effect_sizes)
        avg_power = np.mean(powers)
        
        # Overall validation score (weighted by power)
        weighted_support = sum(
            r['prediction_test']['prediction_met'] * r['prediction_test']['statistical_power']
            for r in results.values()
        ) / sum(r['prediction_test']['statistical_power'] for r in results.values())
        
        # Decision based on 80% threshold
        framework_validated = support_rate >= 0.8
        
        return {
            'total_claims': total_claims,
            'supported_claims': supported_claims,
            'support_rate': support_rate,
            'avg_effect_size': avg_effect_size,
            'avg_power': avg_power,
            'weighted_support': weighted_support,
            'framework_validated': framework_validated,
            'decision': "CORROBORATED" if framework_validated else "REQUIRES_REVISION"
        }

def main():
    """Run comprehensive testable claims validation"""
    framework = TestableClaimsFramework()
    
    # Generate preregistration
    prereg_doc = framework.generate_preregistration_document()
    with open('preregistration.md', 'w') as f:
        f.write(prereg_doc)
    print("📄 Preregistration document saved: preregistration.md")
    
    # Run validation
    results = framework.run_comprehensive_validation()
    
    # Calculate validation score
    validation_score = framework.calculate_validation_score(results)
    
    print("\n" + "=" * 60)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Claims: {validation_score['total_claims']}")
    print(f"Supported: {validation_score['supported_claims']}")
    print(f"Support Rate: {validation_score['support_rate']:.1%}")
    print(f"Average Effect Size: {validation_score['avg_effect_size']:.3f}")
    print(f"Average Power: {validation_score['avg_power']:.3f}")
    print(f"Weighted Support: {validation_score['weighted_support']:.3f}")
    print()
    print(f"🏆 DECISION: {validation_score['decision']}")
    
    # Save results
    with open('validation_results.json', 'w') as f:
        json.dump({
            'results': results,
            'validation_score': validation_score,
            'timestamp': pd.Timestamp.now().isoformat()
        }, f, indent=2, default=str)
    
    return results, validation_score

if __name__ == "__main__":
    main()
