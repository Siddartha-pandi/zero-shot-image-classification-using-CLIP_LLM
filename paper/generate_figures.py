"""
Generate figures for the research paper.

This script creates all necessary figures for the paper including:
1. Adaptation curve (accuracy vs number of samples)
2. Hyperparameter sensitivity plots
3. Domain comparison bar charts
4. Component ablation visualization

Usage:
    python generate_figures.py --results_dir ./results --output_dir ./figures
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def plot_adaptation_curve(results: Dict, output_path: Path):
    """
    Plot accuracy vs number of adaptation samples.
    
    Args:
        results: Dictionary with structure:
            {
                'natural': {'samples': [0, 50, 100, ...], 'accuracy': [68.3, 70.1, ...]},
                'medical': {...},
                ...
            }
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    colors = {
        'natural': '#2E86AB',
        'medical': '#A23B72',
        'satellite': '#F18F01',
        'anime': '#C73E1D',
        'sketch': '#6A994E'
    }
    
    markers = {
        'natural': 'o',
        'medical': 's',
        'satellite': '^',
        'anime': 'D',
        'sketch': 'v'
    }
    
    for domain, data in results.items():
        samples = data['samples']
        accuracy = data['accuracy']
        std = data.get('std', [0] * len(accuracy))
        
        ax.errorbar(
            samples, accuracy, yerr=std,
            marker=markers[domain],
            color=colors[domain],
            label=domain.capitalize(),
            linewidth=2,
            markersize=6,
            capsize=3
        )
    
    ax.set_xlabel('Number of Unlabeled Adaptation Samples')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Accuracy Improvement Through Adaptive Learning')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'adaptation_curve.pdf', bbox_inches='tight')
    plt.savefig(output_path / 'adaptation_curve.png', bbox_inches='tight')
    print(f"✓ Saved adaptation curve to {output_path}")


def plot_hyperparameter_sensitivity(results: Dict, output_path: Path):
    """
    Plot accuracy vs hyperparameters (learning rate and confidence threshold).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Learning rate sensitivity
    alpha_values = results['alpha']['values']
    alpha_accuracy = results['alpha']['accuracy']
    
    ax1.plot(alpha_values, alpha_accuracy, 'o-', linewidth=2, markersize=6, color='#2E86AB')
    ax1.axvline(x=0.05, color='red', linestyle='--', alpha=0.7, label='Default (α=0.05)')
    ax1.set_xlabel('Learning Rate (α)')
    ax1.set_ylabel('Top-1 Accuracy (%)')
    ax1.set_title('Sensitivity to Learning Rate')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Confidence threshold sensitivity
    tau_values = results['tau']['values']
    tau_accuracy = results['tau']['accuracy']
    
    ax2.plot(tau_values, tau_accuracy, 's-', linewidth=2, markersize=6, color='#A23B72')
    ax2.axvline(x=0.15, color='red', linestyle='--', alpha=0.7, label='Default (τ=0.15)')
    ax2.set_xlabel('Confidence Threshold (τ)')
    ax2.set_ylabel('Top-1 Accuracy (%)')
    ax2.set_title('Sensitivity to Confidence Threshold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'hyperparameter_sensitivity.pdf', bbox_inches='tight')
    plt.savefig(output_path / 'hyperparameter_sensitivity.png', bbox_inches='tight')
    print(f"✓ Saved hyperparameter sensitivity to {output_path}")


def plot_domain_comparison(results: Dict, output_path: Path):
    """
    Bar chart comparing generic vs domain-aware prompts.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    domains = list(results.keys())
    generic = [results[d]['generic'] for d in domains]
    domain_aware = [results[d]['domain_aware'] for d in domains]
    
    x = np.arange(len(domains))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, generic, width, label='Generic Prompts', 
                   color='#E0E0E0', edgecolor='black')
    bars2 = ax.bar(x + width/2, domain_aware, width, label='Domain-Aware Prompts',
                   color='#2E86AB', edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)
    
    # Add improvement arrows
    for i, (g, d) in enumerate(zip(generic, domain_aware)):
        improvement = d - g
        if improvement > 0:
            ax.annotate('', xy=(i, d), xytext=(i, g),
                       arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
            ax.text(i + 0.4, (g + d) / 2, f'+{improvement:.1f}%',
                   fontsize=8, color='green', weight='bold')
    
    ax.set_xlabel('Domain')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Impact of Domain-Aware Prompts Across Domains')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in domains], rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / 'domain_comparison.pdf', bbox_inches='tight')
    plt.savefig(output_path / 'domain_comparison.png', bbox_inches='tight')
    print(f"✓ Saved domain comparison to {output_path}")


def plot_ablation_study(results: Dict, output_path: Path):
    """
    Horizontal bar chart showing component contributions.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    components = list(results.keys())
    accuracy = [results[c]['accuracy'] for c in components]
    
    colors = ['#E0E0E0'] + ['#2E86AB'] * (len(components) - 1)
    
    y_pos = np.arange(len(components))
    bars = ax.barh(y_pos, accuracy, color=colors, edgecolor='black')
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracy)):
        width = bar.get_width()
        label = f'{acc:.1f}%'
        if i > 0:
            delta = acc - accuracy[0]
            label += f' (+{delta:.1f}%)'
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
               label, ha='left', va='center', fontsize=9, weight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.replace('_', ' ').title() for c in components])
    ax.set_xlabel('Top-1 Accuracy (%)')
    ax.set_title('Ablation Study: Component Contributions')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim([0, max(accuracy) + 10])
    
    plt.tight_layout()
    plt.savefig(output_path / 'ablation_study.pdf', bbox_inches='tight')
    plt.savefig(output_path / 'ablation_study.png', bbox_inches='tight')
    print(f"✓ Saved ablation study to {output_path}")


def plot_method_comparison(results: Dict, output_path: Path):
    """
    Grouped bar chart comparing different methods across datasets.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    methods = list(results.keys())
    datasets = list(results[methods[0]].keys())
    
    x = np.arange(len(datasets))
    width = 0.15
    
    colors = ['#E0E0E0', '#BDBDBD', '#9E9E9E', '#757575', '#616161', '#2E86AB', '#1565C0']
    
    for i, method in enumerate(methods):
        values = [results[method][ds] for ds in datasets]
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=method, color=colors[i % len(colors)],
               edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Method Comparison Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha='right')
    ax.legend(loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / 'method_comparison.pdf', bbox_inches='tight')
    plt.savefig(output_path / 'method_comparison.png', bbox_inches='tight')
    print(f"✓ Saved method comparison to {output_path}")


def create_example_data():
    """
    Create example data for demonstration purposes.
    Replace this with actual experimental results.
    """
    return {
        'adaptation': {
            'natural': {
                'samples': [0, 50, 100, 200, 300, 500],
                'accuracy': [68.3, 70.1, 71.2, 72.0, 72.5, 73.2],
                'std': [0.5, 0.4, 0.4, 0.3, 0.3, 0.3]
            },
            'medical': {
                'samples': [0, 50, 100, 200, 300, 500],
                'accuracy': [42.1, 45.3, 47.8, 49.5, 50.8, 52.7],
                'std': [0.8, 0.7, 0.6, 0.5, 0.5, 0.4]
            },
            'satellite': {
                'samples': [0, 50, 100, 200, 300, 500],
                'accuracy': [51.2, 54.1, 56.8, 59.2, 61.0, 63.4],
                'std': [0.7, 0.6, 0.5, 0.4, 0.4, 0.3]
            },
            'anime': {
                'samples': [0, 50, 100, 200, 300, 500],
                'accuracy': [65.4, 66.8, 67.9, 68.7, 69.2, 69.8],
                'std': [0.6, 0.5, 0.5, 0.4, 0.4, 0.4]
            },
            'sketch': {
                'samples': [0, 50, 100, 200, 300, 500],
                'accuracy': [58.7, 60.1, 61.3, 62.0, 62.5, 63.1],
                'std': [0.7, 0.6, 0.5, 0.5, 0.4, 0.4]
            }
        },
        'hyperparameters': {
            'alpha': {
                'values': [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.5],
                'accuracy': [69.2, 71.3, 72.5, 72.3, 71.8, 70.5, 67.9]
            },
            'tau': {
                'values': [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
                'accuracy': [70.8, 72.1, 72.5, 72.3, 71.5, 70.2, 68.9]
            }
        },
        'domain': {
            'natural': {'generic': 82.3, 'domain_aware': 83.1},
            'medical': {'generic': 42.1, 'domain_aware': 48.7},
            'satellite': {'generic': 51.2, 'domain_aware': 58.3},
            'anime': {'generic': 65.4, 'domain_aware': 68.9},
            'sketch': {'generic': 58.7, 'domain_aware': 62.3}
        },
        'ablation': {
            'baseline': {'accuracy': 68.3},
            'domain_prompts': {'accuracy': 70.2},
            'domain_prompts_adaptive': {'accuracy': 71.8},
            'domain_prompts_llm': {'accuracy': 71.1},
            'domain_prompts_caption': {'accuracy': 70.6},
            'domain_adaptive_llm': {'accuracy': 72.9},
            'full_system': {'accuracy': 74.1}
        },
        'methods': {
            'CLIP Baseline': {
                'ImageNet': 68.3, 'ChestX-ray': 42.1, 'EuroSAT': 51.2,
                'Oxford Pets': 83.5, 'Food-101': 79.8
            },
            'CLIP Ensemble': {
                'ImageNet': 69.1, 'ChestX-ray': 43.8, 'EuroSAT': 53.6,
                'Oxford Pets': 85.2, 'Food-101': 81.3
            },
            'DCLIP': {
                'ImageNet': 69.8, 'ChestX-ray': 45.2, 'EuroSAT': 54.1,
                'Oxford Pets': 86.1, 'Food-101': 82.1
            },
            'WaffleCLIP': {
                'ImageNet': 70.2, 'ChestX-ray': 44.9, 'EuroSAT': 54.8,
                'Oxford Pets': 86.8, 'Food-101': 82.7
            },
            'AutoCLIP': {
                'ImageNet': 70.9, 'ChestX-ray': 46.3, 'EuroSAT': 56.2,
                'Oxford Pets': 87.9, 'Food-101': 83.4
            },
            'TPT': {
                'ImageNet': 71.4, 'ChestX-ray': 47.1, 'EuroSAT': 57.0,
                'Oxford Pets': 88.5, 'Food-101': 84.2
            },
            'Ours': {
                'ImageNet': 74.1, 'ChestX-ray': 52.7, 'EuroSAT': 63.4,
                'Oxford Pets': 91.8, 'Food-101': 87.3
            }
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Generate figures for research paper')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='./figures',
                       help='Directory to save generated figures')
    parser.add_argument('--use_example_data', action='store_true',
                       help='Use example data instead of loading from files')
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("Generating figures for research paper...")
    print(f"Output directory: {output_path.absolute()}\n")
    
    if args.use_example_data:
        print("Using example data (replace with actual experimental results)")
        data = create_example_data()
    else:
        # Load actual results from files
        results_path = Path(args.results_dir)
        with open(results_path / 'results.json', 'r') as f:
            data = json.load(f)
    
    # Generate all figures
    plot_adaptation_curve(data['adaptation'], output_path)
    plot_hyperparameter_sensitivity(data['hyperparameters'], output_path)
    plot_domain_comparison(data['domain'], output_path)
    plot_ablation_study(data['ablation'], output_path)
    plot_method_comparison(data['methods'], output_path)
    
    print("\n✓ All figures generated successfully!")
    print(f"  - adaptation_curve.pdf")
    print(f"  - hyperparameter_sensitivity.pdf")
    print(f"  - domain_comparison.pdf")
    print(f"  - ablation_study.pdf")
    print(f"  - method_comparison.pdf")
    print(f"\nNote: Also saved PNG versions for preview")


if __name__ == '__main__':
    main()
