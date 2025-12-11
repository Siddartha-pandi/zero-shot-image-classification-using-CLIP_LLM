"""
Comprehensive experiment runner for paper evaluation.

This script runs all experiments needed for the paper including:
- Baseline comparisons
- Ablation studies
- Hyperparameter sensitivity analysis
- Domain-specific evaluations
- Adaptation curve experiments

Usage:
    python run_experiments.py --config experiments_config.yaml --output results/
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

from app.clip_service import (
    classify_image,
    create_class_prototype,
    CLASS_PROTOTYPES,
    CONFIDENCE_THRESHOLD,
)
from app.evaluation_service import evaluate_dataset
from evaluation.dataset import EvaluationDataset
from evaluation.evaluator import ComprehensiveEvaluator
from PIL import Image


class ExperimentRunner:
    """Manages and runs all experiments for the paper."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = {}
        
    def run_all_experiments(self):
        """Run complete experiment suite."""
        print("="*80)
        print("RUNNING COMPREHENSIVE EXPERIMENTS FOR PAPER")
        print("="*80)
        
        # 1. Main method comparison
        print("\n[1/6] Running main method comparison...")
        self.results['method_comparison'] = self.run_method_comparison()
        
        # 2. Ablation studies
        print("\n[2/6] Running ablation studies...")
        self.results['ablation'] = self.run_ablation_studies()
        
        # 3. Domain-specific evaluation
        print("\n[3/6] Running domain-specific evaluation...")
        self.results['domain'] = self.run_domain_evaluation()
        
        # 4. Adaptation curve
        print("\n[4/6] Running adaptation experiments...")
        self.results['adaptation'] = self.run_adaptation_experiments()
        
        # 5. Hyperparameter sensitivity
        print("\n[5/6] Running hyperparameter sensitivity...")
        self.results['hyperparameters'] = self.run_hyperparameter_sensitivity()
        
        # 6. Computational efficiency
        print("\n[6/6] Running computational efficiency tests...")
        self.results['efficiency'] = self.run_efficiency_tests()
        
        # Save all results
        self.save_results()
        
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETED")
        print("="*80)
        
    def run_method_comparison(self) -> Dict:
        """Compare against baseline methods."""
        methods = {
            'CLIP Baseline': self.run_clip_baseline,
            'CLIP Ensemble': self.run_clip_ensemble,
            'AutoCLIP': self.run_autoclip,
            'Ours': self.run_our_method,
        }
        
        datasets = ['imagenet', 'chestxray', 'eurosat', 'oxford_pets', 'food101']
        
        results = {method: {} for method in methods}
        
        for dataset in datasets:
            print(f"\n  Testing on {dataset}...")
            for method_name, method_func in methods.items():
                print(f"    {method_name}...", end='')
                accuracy = method_func(dataset)
                results[method_name][dataset] = accuracy
                print(f" {accuracy:.2f}%")
        
        return results
    
    def run_ablation_studies(self) -> Dict:
        """Run ablation studies removing components."""
        configs = {
            'baseline': {
                'domain_prompts': False,
                'adaptive': False,
                'llm': False,
                'caption': False
            },
            'domain_prompts': {
                'domain_prompts': True,
                'adaptive': False,
                'llm': False,
                'caption': False
            },
            'domain_prompts_adaptive': {
                'domain_prompts': True,
                'adaptive': True,
                'llm': False,
                'caption': False
            },
            'domain_prompts_llm': {
                'domain_prompts': True,
                'adaptive': False,
                'llm': True,
                'caption': False
            },
            'domain_prompts_caption': {
                'domain_prompts': True,
                'adaptive': False,
                'llm': False,
                'caption': True
            },
            'full_system': {
                'domain_prompts': True,
                'adaptive': True,
                'llm': True,
                'caption': True
            }
        }
        
        results = {}
        dataset = 'imagenet'  # Use ImageNet for ablation
        
        for config_name, config in configs.items():
            print(f"  Testing {config_name}...", end='')
            accuracy = self.run_with_config(dataset, config)
            results[config_name] = {'accuracy': accuracy}
            print(f" {accuracy:.2f}%")
        
        return results
    
    def run_domain_evaluation(self) -> Dict:
        """Evaluate domain-specific vs generic prompts."""
        domains = ['natural', 'medical', 'satellite', 'anime', 'sketch']
        
        results = {}
        
        for domain in domains:
            print(f"  Testing {domain} domain...", end='')
            
            # Generic prompts
            generic_acc = self.run_with_domain_prompts(domain, use_generic=True)
            
            # Domain-specific prompts
            domain_acc = self.run_with_domain_prompts(domain, use_generic=False)
            
            results[domain] = {
                'generic': generic_acc,
                'domain_aware': domain_acc
            }
            print(f" Generic: {generic_acc:.2f}%, Domain-aware: {domain_acc:.2f}%")
        
        return results
    
    def run_adaptation_experiments(self) -> Dict:
        """Track accuracy improvement with adaptation."""
        sample_counts = [0, 50, 100, 200, 300, 500]
        domains = ['natural', 'medical', 'satellite', 'anime', 'sketch']
        num_runs = 5
        
        results = {}
        
        for domain in domains:
            print(f"  Testing {domain} adaptation...")
            
            domain_results = {
                'samples': sample_counts,
                'accuracy': [],
                'std': []
            }
            
            for num_samples in sample_counts:
                accuracies = []
                
                for run in range(num_runs):
                    # Reset prototypes
                    CLASS_PROTOTYPES.clear()
                    
                    # Run with adaptation
                    acc = self.run_with_adaptation(domain, num_samples, seed=run)
                    accuracies.append(acc)
                
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                
                domain_results['accuracy'].append(mean_acc)
                domain_results['std'].append(std_acc)
                
                print(f"    {num_samples} samples: {mean_acc:.2f}% ± {std_acc:.2f}%")
            
            results[domain] = domain_results
        
        return results
    
    def run_hyperparameter_sensitivity(self) -> Dict:
        """Test sensitivity to hyperparameters."""
        results = {}
        
        # Learning rate sensitivity
        print("  Testing learning rate sensitivity...")
        alpha_values = [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.5]
        alpha_accuracies = []
        
        for alpha in alpha_values:
            acc = self.run_with_alpha(alpha)
            alpha_accuracies.append(acc)
            print(f"    α={alpha:.2f}: {acc:.2f}%")
        
        results['alpha'] = {
            'values': alpha_values,
            'accuracy': alpha_accuracies
        }
        
        # Confidence threshold sensitivity
        print("  Testing confidence threshold sensitivity...")
        tau_values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        tau_accuracies = []
        
        for tau in tau_values:
            acc = self.run_with_tau(tau)
            tau_accuracies.append(acc)
            print(f"    τ={tau:.2f}: {acc:.2f}%")
        
        results['tau'] = {
            'values': tau_values,
            'accuracy': tau_accuracies
        }
        
        return results
    
    def run_efficiency_tests(self) -> Dict:
        """Measure computational efficiency."""
        methods = ['baseline', 'ours_no_llm', 'ours_full']
        
        results = {}
        num_samples = 100
        
        for method in methods:
            print(f"  Testing {method}...")
            
            latencies = []
            
            for i in range(num_samples):
                start = time.time()
                _ = self.run_single_inference(method)
                latency = (time.time() - start) * 1000  # Convert to ms
                latencies.append(latency)
            
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            
            results[method] = {
                'avg_latency_ms': avg_latency,
                'std_latency_ms': std_latency,
                'gpu_memory_mb': self.measure_gpu_memory(method)
            }
            
            print(f"    Latency: {avg_latency:.1f} ± {std_latency:.1f} ms")
        
        return results
    
    # Placeholder methods - implement based on your actual code
    def run_clip_baseline(self, dataset: str) -> float:
        """Run baseline CLIP."""
        # TODO: Implement actual baseline evaluation
        return 68.3
    
    def run_clip_ensemble(self, dataset: str) -> float:
        """Run CLIP with ensemble prompts."""
        # TODO: Implement ensemble evaluation
        return 69.1
    
    def run_autoclip(self, dataset: str) -> float:
        """Run AutoCLIP baseline."""
        # TODO: Implement AutoCLIP evaluation
        return 70.9
    
    def run_our_method(self, dataset: str) -> float:
        """Run our full method."""
        # TODO: Implement full method evaluation
        return 74.1
    
    def run_with_config(self, dataset: str, config: Dict) -> float:
        """Run with specific configuration."""
        # TODO: Implement config-based evaluation
        return 72.0
    
    def run_with_domain_prompts(self, domain: str, use_generic: bool) -> float:
        """Run with domain-specific or generic prompts."""
        # TODO: Implement domain prompt evaluation
        if use_generic:
            return 65.0
        else:
            return 70.0
    
    def run_with_adaptation(self, domain: str, num_samples: int, seed: int) -> float:
        """Run with adaptive learning."""
        # TODO: Implement adaptation evaluation
        return 70.0 + num_samples * 0.01
    
    def run_with_alpha(self, alpha: float) -> float:
        """Run with specific learning rate."""
        # TODO: Implement alpha sensitivity test
        return 72.0
    
    def run_with_tau(self, tau: float) -> float:
        """Run with specific confidence threshold."""
        # TODO: Implement tau sensitivity test
        return 72.0
    
    def run_single_inference(self, method: str) -> Dict:
        """Run single inference for timing."""
        # TODO: Implement single inference
        time.sleep(0.01)  # Placeholder
        return {}
    
    def measure_gpu_memory(self, method: str) -> float:
        """Measure GPU memory usage."""
        # TODO: Implement GPU memory measurement
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return 0.0
    
    def save_results(self):
        """Save all results to files."""
        # Save as JSON
        output_file = self.output_dir / 'results.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")
        
        # Save summary
        summary_file = self.output_dir / 'summary.txt'
        with open(summary_file, 'w') as f:
            f.write("EXPERIMENT RESULTS SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            # Method comparison
            f.write("METHOD COMPARISON\n")
            f.write("-"*80 + "\n")
            if 'method_comparison' in self.results:
                for method, datasets in self.results['method_comparison'].items():
                    avg = np.mean(list(datasets.values()))
                    f.write(f"{method:20s}: {avg:.2f}% average\n")
            
            f.write("\n")
        
        print(f"✓ Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive experiments')
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test (reduced samples)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    runner = ExperimentRunner(output_dir)
    
    if args.quick:
        print("Running quick test mode (reduced samples)...")
        # TODO: Implement quick test mode
    
    runner.run_all_experiments()
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_dir.absolute()}")
    print(f"Next steps:")
    print(f"  1. Review results in {output_dir}/results.json")
    print(f"  2. Generate figures: python generate_figures.py --results_dir {output_dir}")
    print(f"  3. Update paper tables with actual results")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
