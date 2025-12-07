#!/usr/bin/env python3
"""
Comprehensive Evaluation Script
Runs ablation study testing CLIP, adaptation, and LLM components
"""

import json
import argparse
from pathlib import Path

from evaluation.dataset import EvaluationDataset
from evaluation.evaluator import ComprehensiveEvaluator


def print_summary_table(results: dict):
    """Print ablation study summary table."""
    print(f"\n{'='*80}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*80}")
    print(f"{'Configuration':<40} {'Top-1':<10} {'Top-5':<10} {'Latency (ms)':<15}")
    print(f"{'-'*80}")
    
    configs = [
        ('1. CLIP Baseline', results.get('1_baseline_clip', {})),
        ('2. + Auto-tuning', results.get('3_auto_tuning', {})),
        ('3. + Online Learning', results.get('4_online_learning', {}).get('after_addition', {})),
        ('4. + LLM Reasoning', results.get('5_full_system_llm', {}).get('llm_metrics', {}))
    ]
    
    for name, metrics in configs:
        if metrics:
            top1 = f"{metrics.get('top1_accuracy', 0)*100:.1f}%"
            top5 = f"{metrics.get('top5_accuracy', 0)*100:.1f}%"
            latency = f"{metrics.get('avg_latency_ms', 0):.1f}"
            print(f"{name:<40} {top1:<10} {top5:<10} {latency:<15}")
    
    print(f"{'='*80}\n")


def save_results(results: dict, output_path: str):
    """Save evaluation results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive CLIP-LLM Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full ablation study
  python run_evaluation.py --dataset test_data.json --mode full

  # Quick baseline test
  python run_evaluation.py --dataset test_data.json --mode baseline

  # Test adaptation
  python run_evaluation.py --dataset test_data.json --mode adaptation
        """
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to evaluation dataset JSON file')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output path for results (default: evaluation_results.json)')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'baseline', 'adaptation', 'online', 'llm'],
                       help='Evaluation mode (default: full)')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from: {args.dataset}")
    try:
        dataset = EvaluationDataset(args.dataset)
    except FileNotFoundError:
        print(f"Error: Dataset file not found: {args.dataset}")
        print("Please create a JSON file with format:")
        print('[{"path": "image.jpg", "label": "cat", "domain": "natural"}, ...]')
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print(f"✓ Loaded {len(dataset)} samples")
    print(f"  Domains: {dataset.get_domains()}")
    print(f"  Classes: {len(dataset.get_labels())} unique labels")
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator(dataset)
    
    if args.mode == 'full':
        print("\n🚀 Running FULL ABLATION STUDY")
        print("This will test 4 configurations:")
        print("  1. CLIP Baseline")
        print("  2. + Auto-tuning")
        print("  3. + Online Learning")
        print("  4. + LLM Reasoning (limited samples)")
        
        results = evaluator.run_full_ablation()
        print_summary_table(results)
        
    elif args.mode == 'baseline':
        evaluator.initialize_classes()
        results = {'baseline': evaluator.run_baseline_clip()}
        
    elif args.mode == 'adaptation':
        evaluator.initialize_classes()
        metrics, curve = evaluator.run_with_adaptation_tracking()
        results = {'adaptation': metrics, 'adaptation_curve': curve}
        
    elif args.mode == 'online':
        results = {'online': evaluator.run_online_learning_experiment()}
        
    elif args.mode == 'llm':
        evaluator.initialize_classes()
        results = {'llm': evaluator.run_with_llm(max_samples=50)}
    
    # Save results
    save_results(results, args.output)
    
    print("\n✅ Evaluation complete!")
    print(f"\nNext steps:")
    print(f"  1. View results: cat {args.output}")
    print(f"  2. Generate plots (requires matplotlib):")
    print(f"     python -c 'import json; print(json.dumps(json.load(open(\"{args.output}\")), indent=2))'")


if __name__ == '__main__':
    main()
