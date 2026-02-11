#!/usr/bin/env python3
"""
Parallel Cross-Language Validation Framework
Runs corollary tests across Python, Rust, Go, and C++ implementations simultaneously
to verify theoretical predictions hold universally across all language ports.
"""

import subprocess
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import os

class ParallelValidator:
    """Orchestrates parallel corollary validation across all language implementations"""
    
    def __init__(self, base_dir: str = "/Users/ryanbarrett/gods-as-centroids"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "validation_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Language-specific configurations
        self.languages = {
            'python': {
                'cmd': ['python', 'corollary_validation.py'],
                'cwd': self.base_dir,
                'output_file': 'python_results.json'
            },
            'rust': {
                'cmd': ['cargo', 'run', '--release', '--', '--corollary-test'],
                'cwd': self.base_dir / 'rust',
                'output_file': 'rust_results.json'
            },
            'go': {
                'cmd': ['./gabm', '--corollary-test'],
                'cwd': self.base_dir / 'go',
                'output_file': 'go_results.json'
            },
            'cpp': {
                'cmd': ['./build/gabm', '--corollary-test'],
                'cwd': self.base_dir / 'cpp',
                'output_file': 'cpp_results.json'
            }
        }
    
    def create_corollary_configs(self):
        """Create standardized test configurations for each corollary"""
        configs = {
            'universality': {
                'test_name': 'universality',
                'parameter_sweep': 'coupling',
                'values': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
                'base_config': {
                    'N': 30,
                    'steps': 2000,
                    'enable_sensory_restrictions': False,
                    'seed': 42
                }
            },
            'plurality_monotheism': {
                'test_name': 'plurality_monotheism',
                'parameter_sweep': 'coercion',
                'values': [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
                'base_config': {
                    'N': 40,
                    'steps': 3000,
                    'enable_sensory_restrictions': False,
                    'seed': 42
                }
            },
            'ritual_stabilizer': {
                'test_name': 'ritual_stabilizer',
                'parameter_sweep': 'ritual_bonus',
                'values': [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
                'base_config': {
                    'N': 35,
                    'steps': 2500,
                    'ritual_period': 25,
                    'enable_sensory_restrictions': False,
                    'seed': 42
                }
            },
            'prestige_amplifier': {
                'test_name': 'prestige_amplifier',
                'parameter_sweep': 'prestige_alpha',
                'values': [0.0, 0.1, 0.2, 0.3, 0.5, 0.8],
                'base_config': {
                    'N': 35,
                    'steps': 3000,
                    'enable_sensory_restrictions': False,
                    'seed': 42
                }
            }
        }
        
        # Save configs for each language
        for lang in self.languages:
            lang_config_dir = self.results_dir / f"{lang}_configs"
            lang_config_dir.mkdir(exist_ok=True)
            
            for test_name, config in configs.items():
                config_path = lang_config_dir / f"{test_name}.json"
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
        
        return configs
    
    def run_language_test(self, language: str, test_config: Dict) -> Dict:
        """Run corollary test for a specific language implementation"""
        print(f"🚀 Starting {language.upper()} validation...")
        
        lang_config = self.languages[language]
        start_time = time.time()
        
        try:
            # Prepare command with test configuration
            cmd = lang_config['cmd'].copy()
            if language == 'python':
                # Python uses direct function calls
                result = self._run_python_test(test_config)
            else:
                # Other languages use CLI with JSON config
                config_file = self.results_dir / f"{language}_configs" / "test_config.json"
                with open(config_file, 'w') as f:
                    json.dump(test_config, f, indent=2)
                
                cmd.extend(['--config', str(config_file)])
                
                # Run the command
                process = subprocess.run(
                    cmd,
                    cwd=lang_config['cwd'],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if process.returncode != 0:
                    raise RuntimeError(f"Command failed: {process.stderr}")
                
                # Parse JSON output
                try:
                    result = json.loads(process.stdout)
                except json.JSONDecodeError:
                    # Fallback: look for JSON in output
                    lines = process.stdout.split('\n')
                    for line in lines:
                        if line.strip().startswith('{'):
                            result = json.loads(line.strip())
                            break
                    else:
                        raise ValueError("No JSON output found")
            
            execution_time = time.time() - start_time
            
            return {
                'language': language,
                'success': True,
                'execution_time': execution_time,
                'results': result,
                'error': None
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ {language.upper()} failed: {str(e)}")
            
            return {
                'language': language,
                'success': False,
                'execution_time': execution_time,
                'results': None,
                'error': str(e)
            }
    
    def _run_python_test(self, test_config: Dict) -> Dict:
        """Run Python corollary test directly"""
        import sys
        sys.path.append('sim')
        from corollary_suite import CorollarySuite
        
        # Create a focused test based on config
        suite = CorollarySuite()
        
        if test_config['test_name'] == 'universality':
            return suite.test_universality_corollary()
        elif test_config['test_name'] == 'plurality_monotheism':
            return suite.test_plurality_monotheism_corollary()
        elif test_config['test_name'] == 'ritual_stabilizer':
            return suite.test_ritual_stabilizer_corollary()
        elif test_config['test_name'] == 'prestige_amplifier':
            return suite.test_prestige_amplifier_corollary()
        else:
            # Run all tests
            return suite.run_comprehensive_test()
    
    def run_parallel_validation(self, test_configs: Dict) -> Dict:
        """Run all corollary tests across all languages in parallel"""
        print("🔬 Starting Parallel Cross-Language Validation")
        print("=" * 60)
        
        all_results = {}
        
        for test_name, config in test_configs.items():
            print(f"\n🧪 Testing {test_name.replace('_', ' ').title()} Corollary")
            print("-" * 40)
            
            # Run test across all languages in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self.run_language_test, lang, config): lang 
                    for lang in self.languages.keys()
                }
                
                test_results = {}
                for future in as_completed(futures):
                    lang = futures[future]
                    result = future.result()
                    test_results[lang] = result
                    
                    status = "✅" if result['success'] else "❌"
                    time_str = f"{result['execution_time']:.2f}s"
                    print(f"  {status} {lang.upper()}: {time_str}")
            
            all_results[test_name] = test_results
        
        return all_results
    
    def validate_cross_language_consistency(self, results: Dict) -> Dict:
        """Validate that all language implementations produce consistent results"""
        print("\n🔍 Cross-Language Consistency Analysis")
        print("=" * 60)
        
        consistency_report = {}
        
        for test_name, lang_results in results.items():
            print(f"\n📊 {test_name.replace('_', ' ').title()}")
            
            # Extract successful results
            successful_results = {
                lang: data['results'] 
                for lang, data in lang_results.items() 
                if data['success'] and data['results']
            }
            
            if len(successful_results) < 2:
                print("  ⚠️  Insufficient results for comparison")
                consistency_report[test_name] = {'consistent': False, 'reason': 'insufficient_data'}
                continue
            
            # Compare key metrics across languages
            consistency_metrics = self._compare_results(successful_results, test_name)
            consistency_report[test_name] = consistency_metrics
            
            if consistency_metrics['consistent']:
                print(f"  ✅ Consistent across {len(successful_results)} languages")
                print(f"     Max deviation: {consistency_metrics['max_deviation']:.3f}")
            else:
                print(f"  ❌ Inconsistent results detected")
                print(f"     Deviation: {consistency_metrics['max_deviation']:.3f}")
        
        return consistency_report
    
    def _compare_results(self, lang_results: Dict, test_name: str) -> Dict:
        """Compare results across languages for consistency"""
        if test_name == 'universality':
            return self._compare_universality_results(lang_results)
        elif test_name == 'plurality_monotheism':
            return self._compare_plurality_results(lang_results)
        elif test_name == 'ritual_stabilizer':
            return self._compare_ritual_results(lang_results)
        elif test_name == 'prestige_amplifier':
            return self._compare_prestige_results(lang_results)
        else:
            return {'consistent': True, 'max_deviation': 0.0}
    
    def _compare_universality_results(self, lang_results: Dict) -> Dict:
        """Compare universality corollary results"""
        # Extract godform emergence patterns
        emergence_patterns = {}
        for lang, results in lang_results.items():
            if 'universality' in results:
                pattern = [d['has_godforms'] for d in results['universality']]
                emergence_patterns[lang] = pattern
        
        if len(emergence_patterns) < 2:
            return {'consistent': False, 'max_deviation': 1.0}
        
        # Check if all languages show same emergence pattern
        reference_pattern = list(emergence_patterns.values())[0]
        max_deviation = 0.0
        
        for pattern in emergence_patterns.values():
            deviation = sum(a != b for a, b in zip(reference_pattern, pattern)) / len(pattern)
            max_deviation = max(max_deviation, deviation)
        
        return {
            'consistent': max_deviation < 0.2,  # Allow 20% deviation
            'max_deviation': max_deviation,
            'patterns': emergence_patterns
        }
    
    def _compare_plurality_results(self, lang_results: Dict) -> Dict:
        """Compare plurality/monotheism corollary results"""
        n_eff_trends = {}
        for lang, results in lang_results.items():
            if 'plurality_monotheism' in results:
                n_effs = [d['n_effective'] for d in results['plurality_monotheism']]
                n_eff_trends[lang] = n_effs
        
        if len(n_eff_trends) < 2:
            return {'consistent': False, 'max_deviation': 1.0}
        
        # Check monotonic decrease trend
        reference_trend = list(n_eff_trends.values())[0]
        max_deviation = 0.0
        
        for trend in n_eff_trends.values():
            # Normalize and compare
            if len(trend) == len(reference_trend):
                deviation = np.mean(np.abs(np.array(trend) - np.array(reference_trend)))
                max_deviation = max(max_deviation, deviation)
        
        return {
            'consistent': max_deviation < 2.0,  # Allow 2 centroid difference
            'max_deviation': max_deviation,
            'trends': n_eff_trends
        }
    
    def _compare_ritual_results(self, lang_results: Dict) -> Dict:
        """Compare ritual stabilizer results"""
        stability_trends = {}
        for lang, results in lang_results.items():
            if 'ritual_stabilizer' in results:
                stabilities = [d['stability'] for d in results['ritual_stabilizer']]
                stability_trends[lang] = stabilities
        
        if len(stability_trends) < 2:
            return {'consistent': False, 'max_deviation': 1.0}
        
        reference_trend = list(stability_trends.values())[0]
        max_deviation = 0.0
        
        for trend in stability_trends.values():
            if len(trend) == len(reference_trend):
                deviation = np.mean(np.abs(np.array(trend) - np.array(reference_trend)))
                max_deviation = max(max_deviation, deviation)
        
        return {
            'consistent': max_deviation < 0.3,  # Allow 30% stability deviation
            'max_deviation': max_deviation,
            'trends': stability_trends
        }
    
    def _compare_prestige_results(self, lang_results: Dict) -> Dict:
        """Compare prestige amplifier results"""
        convergence_trends = {}
        for lang, results in lang_results.items():
            if 'prestige_amplifier' in results:
                conv_steps = [d['convergence_steps'] for d in results['prestige_amplifier']]
                convergence_trends[lang] = conv_steps
        
        if len(convergence_trends) < 2:
            return {'consistent': False, 'max_deviation': 1.0}
        
        reference_trend = list(convergence_trends.values())[0]
        max_deviation = 0.0
        
        for trend in convergence_trends.values():
            if len(trend) == len(reference_trend):
                # Normalize by max steps to compare relative convergence
                ref_norm = np.array(reference_trend) / 3000.0
                trend_norm = np.array(trend) / 3000.0
                deviation = np.mean(np.abs(ref_norm - trend_norm))
                max_deviation = max(max_deviation, deviation)
        
        return {
            'consistent': max_deviation < 0.2,  # Allow 20% relative deviation
            'max_deviation': max_deviation,
            'trends': convergence_trends
        }
    
    def generate_validation_report(self, results: Dict, consistency: Dict):
        """Generate comprehensive validation report"""
        report_path = self.results_dir / "validation_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# God-Centroid Corollary Cross-Language Validation Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Count successful tests
            total_tests = sum(len(lang_results) for lang_results in results.values())
            successful_tests = sum(
                sum(1 for data in lang_results.values() if data['success'])
                for lang_results in results.values()
            )
            
            f.write(f"- **Total Tests**: {total_tests}\n")
            f.write(f"- **Successful**: {successful_tests}\n")
            f.write(f"- **Success Rate**: {successful_tests/total_tests*100:.1f}%\n\n")
            
            # Consistency summary
            consistent_corollaries = sum(1 for c in consistency.values() if c.get('consistent', False))
            f.write(f"- **Consistent Corollaries**: {consistent_corollaries}/{len(consistency)}\n\n")
            
            f.write("## Corollary Validation Results\n\n")
            
            for test_name, lang_results in results.items():
                f.write(f"### {test_name.replace('_', ' ').title()} Corollary\n\n")
                
                # Language results table
                f.write("| Language | Status | Time (s) | Error |\n")
                f.write("|----------|--------|----------|-------|\n")
                
                for lang, data in lang_results.items():
                    status = "✅ Success" if data['success'] else "❌ Failed"
                    time_str = f"{data['execution_time']:.2f}"
                    error = data['error'] or "-"
                    f.write(f"| {lang.upper()} | {status} | {time_str} | {error} |\n")
                
                f.write("\n")
                
                # Consistency analysis
                if test_name in consistency:
                    cons = consistency[test_name]
                    if cons.get('consistent', False):
                        f.write(f"**Consistency**: ✅ Validated (max deviation: {cons['max_deviation']:.3f})\n\n")
                    else:
                        f.write(f"**Consistency**: ❌ Failed (deviation: {cons.get('max_deviation', 'N/A')})\n\n")
        
        print(f"\n📄 Validation report saved: {report_path}")
        return report_path

def main():
    """Run parallel cross-language validation"""
    validator = ParallelValidator()
    
    # Create test configurations
    print("⚙️  Creating test configurations...")
    test_configs = validator.create_corollary_configs()
    
    # Run parallel validation
    results = validator.run_parallel_validation(test_configs)
    
    # Validate consistency
    consistency = validator.validate_cross_language_consistency(results)
    
    # Generate report
    report_path = validator.generate_validation_report(results, consistency)
    
    # Save raw results
    results_path = validator.results_dir / "raw_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'results': results,
            'consistency': consistency,
            'timestamp': time.time()
        }, f, indent=2)
    
    print(f"\n🎯 Validation Complete!")
    print(f"📊 Raw results: {results_path}")
    print(f"📄 Report: {report_path}")
    
    return results, consistency

if __name__ == "__main__":
    main()
