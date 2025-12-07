# Visualization and Reporting Tools
# Generate plots, confusion matrices, and formatted reports

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def plot_ablation_comparison(results: Dict, output_path: str = 'ablation_comparison.png'):
    """
    Create bar chart comparing different configurations.
    """
    configs = []
    top1_scores = []
    top5_scores = []
    
    # Extract data
    if '1_baseline_clip' in results:
        configs.append('CLIP\nBaseline')
        top1_scores.append(results['1_baseline_clip'].get('top1_accuracy', 0) * 100)
        top5_scores.append(results['1_baseline_clip'].get('top5_accuracy', 0) * 100)
    
    if '2_domain_prompts' in results:
        configs.append('+ Domain\nPrompts')
        top1_scores.append(results['2_domain_prompts'].get('top1_accuracy', 0) * 100)
        top5_scores.append(results['2_domain_prompts'].get('top5_accuracy', 0) * 100)
    
    if '3_auto_tuning' in results:
        configs.append('+ Auto-\nTuning')
        top1_scores.append(results['3_auto_tuning'].get('top1_accuracy', 0) * 100)
        top5_scores.append(results['3_auto_tuning'].get('top5_accuracy', 0) * 100)
    
    if '4_online_learning' in results and 'after_addition' in results['4_online_learning']:
        configs.append('+ Online\nLearning')
        top1_scores.append(results['4_online_learning']['after_addition'].get('top1_accuracy', 0) * 100)
        top5_scores.append(results['4_online_learning']['after_addition'].get('top5_accuracy', 0) * 100)
    
    if '5_full_system_llm' in results and 'llm_metrics' in results['5_full_system_llm']:
        configs.append('+ LLM\nReasoning')
        top1_scores.append(results['5_full_system_llm']['llm_metrics'].get('top1_accuracy', 0) * 100)
        top5_scores.append(results['5_full_system_llm']['llm_metrics'].get('top5_accuracy', 0) * 100)
    
    # Create plot
    x = np.arange(len(configs))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, top1_scores, width, label='Top-1 Accuracy', color='#3b82f6')
    bars2 = ax.bar(x + width/2, top5_scores, width, label='Top-5 Accuracy', color='#10b981')
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Progressive Feature Addition', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Ablation comparison plot saved to: {output_path}")
    plt.close()


def plot_adaptation_curve(accuracy_curve: List[float], output_path: str = 'adaptation_curve.png'):
    """
    Plot accuracy over time during adaptation.
    """
    if not accuracy_curve:
        print("No adaptation curve data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = np.arange(1, len(accuracy_curve) + 1) * 50  # Assuming window size of 50
    ax.plot(steps, [a * 100 for a in accuracy_curve], 
            marker='o', linewidth=2, markersize=6, color='#3b82f6')
    
    ax.set_xlabel('Number of Images Processed', fontsize=12, fontweight='bold')
    ax.set_ylabel('Window Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Improvement During Auto-Tuning', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(steps, [a * 100 for a in accuracy_curve], 1)
    p = np.poly1d(z)
    ax.plot(steps, p(steps), "--", color='#ef4444', alpha=0.8, label='Trend')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Adaptation curve plot saved to: {output_path}")
    plt.close()


def plot_domain_performance(results: Dict, output_path: str = 'domain_performance.png'):
    """
    Plot per-domain accuracy comparison.
    """
    # Extract domain accuracies from best configuration
    domain_acc = None
    
    if '5_full_system_llm' in results and 'llm_metrics' in results['5_full_system_llm']:
        domain_acc = results['5_full_system_llm']['llm_metrics'].get('domain_accuracy', {})
    elif '3_auto_tuning' in results:
        domain_acc = results['3_auto_tuning'].get('domain_accuracy', {})
    
    if not domain_acc:
        print("No domain performance data available")
        return
    
    domains = list(domain_acc.keys())
    accuracies = [domain_acc[d] * 100 for d in domains]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(domains, accuracies, color='#8b5cf6', alpha=0.8)
    
    ax.set_xlabel('Domain', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Domain Classification Accuracy', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=10)
    
    # Add average line
    avg = np.mean(accuracies)
    ax.axhline(y=avg, color='#ef4444', linestyle='--', linewidth=2, label=f'Average: {avg:.1f}%')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Domain performance plot saved to: {output_path}")
    plt.close()


def plot_confusion_matrix(predictions: List[str], ground_truths: List[str], 
                         labels: List[str], output_path: str = 'confusion_matrix.png'):
    """
    Create confusion matrix heatmap.
    """
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(ground_truths, predictions, labels=labels)
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    # Create plot
    fig, ax = plt.subplots(figsize=(max(10, len(labels)), max(8, len(labels))))
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={'label': 'Normalized Frequency'})
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {output_path}")
    plt.close()


def plot_online_learning_comparison(results: Dict, output_path: str = 'online_learning.png'):
    """
    Show before/after comparison for online learning.
    """
    online_results = results.get('4_online_learning', {})
    
    if not online_results:
        print("No online learning data available")
        return
    
    before = online_results.get('before_addition', {}).get('top1_accuracy', 0) * 100
    after = online_results.get('after_addition', {}).get('top1_accuracy', 0) * 100
    improvement = online_results.get('improvement', 0) * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Before\nAdding Classes', 'After\nAdding Classes']
    values = [before, after]
    colors = ['#ef4444', '#10b981']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, width=0.6)
    
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Online Learning: New Class Addition Impact', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add improvement annotation
    ax.annotate(f'+{improvement:.1f}%',
                xy=(1, after), xytext=(0.5, (before + after) / 2),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                fontsize=14, fontweight='bold', color='#10b981')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Online learning comparison saved to: {output_path}")
    plt.close()


def generate_latex_table(results: Dict, output_path: str = 'results_table.tex'):
    """
    Generate LaTeX table for research paper/report.
    """
    latex = []
    latex.append(r'\begin{table}[h]')
    latex.append(r'\centering')
    latex.append(r'\caption{Ablation Study Results}')
    latex.append(r'\label{tab:ablation}')
    latex.append(r'\begin{tabular}{lccc}')
    latex.append(r'\hline')
    latex.append(r'Configuration & Top-1 Acc. & Top-5 Acc. & Latency (ms) \\')
    latex.append(r'\hline')
    
    configs = [
        ('CLIP Baseline', results.get('1_baseline_clip', {})),
        ('+ Domain Prompts', results.get('2_domain_prompts', {})),
        ('+ Auto-Tuning', results.get('3_auto_tuning', {})),
        ('+ Online Learning', results.get('4_online_learning', {}).get('after_addition', {})),
        ('+ LLM Reasoning', results.get('5_full_system_llm', {}).get('llm_metrics', {}))
    ]
    
    for name, metrics in configs:
        if metrics:
            top1 = metrics.get('top1_accuracy', 0) * 100
            top5 = metrics.get('top5_accuracy', 0) * 100
            latency = metrics.get('avg_latency_ms', 0)
            latex.append(f'{name} & {top1:.1f}\\% & {top5:.1f}\\% & {latency:.1f} \\\\')
    
    latex.append(r'\hline')
    latex.append(r'\end{tabular}')
    latex.append(r'\end{table}')
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"✓ LaTeX table saved to: {output_path}")


def generate_markdown_report(results: Dict, output_path: str = 'EVALUATION_REPORT.md'):
    """
    Generate comprehensive markdown report.
    """
    report = []
    report.append('# Comprehensive Evaluation Report\n')
    report.append(f'Generated: {Path(output_path).stem}\n')
    report.append('---\n')
    
    # Executive Summary
    report.append('## Executive Summary\n')
    best_acc = 0
    best_config = ''
    
    for key, val in results.items():
        if isinstance(val, dict) and 'top1_accuracy' in val:
            acc = val['top1_accuracy']
            if acc > best_acc:
                best_acc = acc
                best_config = key
    
    report.append(f'**Best Configuration:** {best_config}\n')
    report.append(f'**Best Top-1 Accuracy:** {best_acc*100:.2f}%\n\n')
    
    # Ablation Study
    report.append('## Ablation Study Results\n')
    report.append('| Configuration | Top-1 Accuracy | Top-5 Accuracy | Avg Latency (ms) |\n')
    report.append('|--------------|----------------|----------------|------------------|\n')
    
    configs = [
        ('1. CLIP Baseline', '1_baseline_clip'),
        ('2. + Domain Prompts', '2_domain_prompts'),
        ('3. + Auto-Tuning', '3_auto_tuning'),
        ('4. + Online Learning', '4_online_learning'),
        ('5. + LLM Reasoning', '5_full_system_llm')
    ]
    
    for name, key in configs:
        metrics = results.get(key, {})
        if key == '4_online_learning':
            metrics = metrics.get('after_addition', {})
        elif key == '5_full_system_llm':
            metrics = metrics.get('llm_metrics', {})
        
        if metrics:
            top1 = metrics.get('top1_accuracy', 0) * 100
            top5 = metrics.get('top5_accuracy', 0) * 100
            latency = metrics.get('avg_latency_ms', 0)
            report.append(f'| {name} | {top1:.1f}% | {top5:.1f}% | {latency:.1f} |\n')
    
    report.append('\n')
    
    # Domain Performance
    report.append('## Per-Domain Performance\n')
    domain_acc = results.get('3_auto_tuning', {}).get('domain_accuracy', {})
    if domain_acc:
        report.append('| Domain | Accuracy |\n')
        report.append('|--------|----------|\n')
        for domain, acc in sorted(domain_acc.items(), key=lambda x: -x[1]):
            report.append(f'| {domain.capitalize()} | {acc*100:.1f}% |\n')
        report.append('\n')
    
    # Online Learning
    report.append('## Online Learning Results\n')
    online = results.get('4_online_learning', {})
    if online:
        before = online.get('before_addition', {}).get('top1_accuracy', 0) * 100
        after = online.get('after_addition', {}).get('top1_accuracy', 0) * 100
        improvement = online.get('improvement', 0) * 100
        
        report.append(f'- **Before adding new classes:** {before:.1f}%\n')
        report.append(f'- **After adding new classes:** {after:.1f}%\n')
        report.append(f'- **Improvement:** +{improvement:.1f}%\n\n')
    
    # LLM Analysis
    report.append('## LLM Reasoning Analysis\n')
    llm_results = results.get('5_full_system_llm', {})
    if llm_results:
        improvements = llm_results.get('llm_improvements', 0)
        report.append(f'- **Cases where LLM corrected CLIP mistakes:** {improvements}\n')
        
        examples = llm_results.get('improvement_examples', [])
        if examples:
            report.append('\n### Example Improvements\n')
            for i, ex in enumerate(examples[:3], 1):
                report.append(f'\n**Example {i}:**\n')
                report.append(f'- True Label: `{ex["true_label"]}`\n')
                report.append(f'- CLIP Prediction: `{ex["clip_label"]}`\n')
                report.append(f'- LLM Correction: `{ex["llm_label"]}`\n')
                report.append(f'- Reasoning: {ex.get("reasoning", "N/A")}\n')
    
    report.append('\n---\n')
    report.append('## Conclusion\n')
    report.append('This evaluation demonstrates the effectiveness of combining CLIP with ')
    report.append('domain-aware prompts, auto-tuning, online learning, and LLM reasoning.\n')
    
    with open(output_path, 'w') as f:
        f.write(''.join(report))
    
    print(f"✓ Markdown report saved to: {output_path}")


def generate_all_visualizations(results_path: str, output_dir: str = 'evaluation_outputs'):
    """
    Generate all plots and reports from evaluation results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"\nGenerating visualizations in: {output_path}")
    print(f"{'='*60}")
    
    # Generate plots
    plot_ablation_comparison(results, output_path / 'ablation_comparison.png')
    
    if '3_adaptation_curve' in results:
        plot_adaptation_curve(results['3_adaptation_curve'], 
                            output_path / 'adaptation_curve.png')
    
    plot_domain_performance(results, output_path / 'domain_performance.png')
    plot_online_learning_comparison(results, output_path / 'online_learning.png')
    
    # Generate reports
    generate_latex_table(results, output_path / 'results_table.tex')
    generate_markdown_report(results, output_path / 'EVALUATION_REPORT.md')
    
    print(f"{'='*60}")
    print(f"✓ All visualizations generated successfully!")
    print(f"\nFiles created in {output_path}:")
    for file in sorted(output_path.iterdir()):
        print(f"  - {file.name}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Evaluation Visualizations')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to evaluation results JSON')
    parser.add_argument('--output-dir', type=str, default='evaluation_outputs',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    generate_all_visualizations(args.results, args.output_dir)
