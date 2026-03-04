"""
Visualization Script for Reference Predictions
Generates charts and insights from the reference predictions dataset
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_reference_predictions():
    """Load the reference predictions JSON file"""
    file_path = Path(__file__).parent / "reference_predictions.json"
    with open(file_path, 'r') as f:
        return json.load(f)


def plot_confidence_distribution(data):
    """Plot confidence score distribution across all predictions"""
    examples = data['examples']
    confidences = [ex['confidence'] for ex in examples]
    domains = [ex['domain'] for ex in examples]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Confidence by domain
    medical_conf = [ex['confidence'] for ex in examples if ex['domain'] == 'Medical Imaging']
    industrial_conf = [ex['confidence'] for ex in examples if ex['domain'] == 'Industrial Inspection']
    
    positions = [1, 2]
    box_data = [medical_conf, industrial_conf]
    bp = ax1.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                     labels=['Medical\nImaging', 'Industrial\nInspection'])
    
    # Color boxes
    colors = ['#4CAF50', '#2196F3']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Confidence Score', fontsize=12)
    ax1.set_title('Confidence Distribution by Domain', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0.85, 0.95)
    
    # Plot 2: Individual predictions
    x_medical = list(range(len(medical_conf)))
    x_industrial = list(range(len(medical_conf), len(medical_conf) + len(industrial_conf)))
    
    ax2.scatter(x_medical, medical_conf, s=150, c='#4CAF50', alpha=0.7, label='Medical (MedCLIP)', edgecolors='black', linewidth=1.5)
    ax2.scatter(x_industrial, industrial_conf, s=150, c='#2196F3', alpha=0.7, label='Industrial (ViT-H/14)', edgecolors='black', linewidth=1.5)
    
    # Add horizontal line for average
    avg_conf = np.mean(confidences)
    ax2.axhline(y=avg_conf, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_conf:.3f}')
    
    ax2.set_xlabel('Prediction Index', fontsize=12)
    ax2.set_ylabel('Confidence Score', fontsize=12)
    ax2.set_title('Individual Prediction Confidence', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0.85, 0.95)
    
    plt.tight_layout()
    plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: confidence_analysis.png")
    plt.close()


def plot_top_predictions_scores(data):
    """Plot top-3 prediction scores for each example"""
    examples = data['examples']
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Top-3 Prediction Scores by Example', fontsize=16, fontweight='bold', y=1.02)
    
    for idx, (ex, ax) in enumerate(zip(examples, axes.flat)):
        top_preds = ex['top_predictions']
        labels = [p['label'] for p in top_preds]
        scores = [p['score'] for p in top_preds]
        
        # Truncate long labels
        labels = [l[:20] + '...' if len(l) > 20 else l for l in labels]
        
        # Color by domain
        color = '#4CAF50' if ex['domain'] == 'Medical Imaging' else '#2196F3'
        
        bars = ax.barh(labels, scores, color=color, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add score labels
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.2f}', ha='left', va='center', fontsize=9)
        
        ax.set_xlim(0, 1.0)
        ax.set_xlabel('Score', fontsize=9)
        ax.set_title(f"{ex['prediction']}\n({ex['domain'][:3]})", fontsize=10, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('top_predictions_breakdown.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: top_predictions_breakdown.png")
    plt.close()


def plot_model_comparison(data):
    """Compare MedCLIP vs ViT-H/14 performance"""
    examples = data['examples']
    
    # Group by model
    medclip = [ex for ex in examples if ex['model_used'] == 'MedCLIP']
    vith14 = [ex for ex in examples if ex['model_used'] == 'CLIP ViT-H/14']
    
    # Compute metrics
    metrics = {
        'MedCLIP': {
            'avg_conf': np.mean([ex['confidence'] for ex in medclip]),
            'min_conf': np.min([ex['confidence'] for ex in medclip]),
            'max_conf': np.max([ex['confidence'] for ex in medclip]),
            'count': len(medclip)
        },
        'ViT-H/14': {
            'avg_conf': np.mean([ex['confidence'] for ex in vith14]),
            'min_conf': np.min([ex['confidence'] for ex in vith14]),
            'max_conf': np.max([ex['confidence'] for ex in vith14]),
            'count': len(vith14)
        }
    }
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Average confidence comparison
    models = list(metrics.keys())
    avg_confs = [metrics[m]['avg_conf'] for m in models]
    colors_list = ['#4CAF50', '#2196F3']
    
    bars = ax1.bar(models, avg_confs, color=colors_list, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, avg_confs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Average Confidence', fontsize=12)
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0.85, 0.92)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Min-Max range
    for i, model in enumerate(models):
        min_val = metrics[model]['min_conf']
        max_val = metrics[model]['max_conf']
        avg_val = metrics[model]['avg_conf']
        
        # Plot range line
        ax2.plot([i, i], [min_val, max_val], 'o-', color=colors_list[i], 
                linewidth=3, markersize=8, alpha=0.7, label=model)
        # Plot average point
        ax2.scatter([i], [avg_val], s=200, c=colors_list[i], 
                   edgecolors='red', linewidth=2, zorder=5, marker='D')
    
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models)
    ax2.set_ylabel('Confidence Score', fontsize=12)
    ax2.set_title('Confidence Range (◆ = Average)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0.85, 0.95)
    ax2.grid(alpha=0.3)
    ax2.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: model_comparison.png")
    plt.close()


def plot_prediction_categories(data):
    """Plot distribution of prediction categories"""
    examples = data['examples']
    
    # Count predictions by category
    predictions = [ex['prediction'] for ex in examples]
    unique_preds, counts = np.unique(predictions, return_counts=True)
    
    # Sort by count
    sorted_indices = np.argsort(counts)[::-1]
    unique_preds = unique_preds[sorted_indices]
    counts = counts[sorted_indices]
    
    # Assign colors by domain
    colors_map = []
    for pred in unique_preds:
        # Find the domain for this prediction
        pred_domain = [ex['domain'] for ex in examples if ex['prediction'] == pred][0]
        color = '#4CAF50' if pred_domain == 'Medical Imaging' else '#2196F3'
        colors_map.append(color)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(unique_preds, counts, color=colors_map, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add count labels
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        ax.text(width + 0.05, bar.get_y() + bar.get_height()/2,
               f'{count}', ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Count', fontsize=12)
    ax.set_title('Prediction Category Distribution', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(counts) + 0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', alpha=0.7, edgecolor='black', label='Medical Imaging'),
        Patch(facecolor='#2196F3', alpha=0.7, edgecolor='black', label='Industrial Inspection')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_categories.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: prediction_categories.png")
    plt.close()


def generate_summary_stats(data):
    """Print summary statistics"""
    examples = data['examples']
    analysis = data['analysis']
    
    print("\n" + "="*70)
    print(" REFERENCE PREDICTIONS SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   • Total Examples: {len(examples)}")
    print(f"   • Domains: {', '.join(data['domains'])}")
    print(f"   • Models: {', '.join(data['models_used'])}")
    
    print(f"\n📈 Confidence Metrics:")
    print(f"   • Overall Average: {analysis['average_confidence']['overall']:.3f}")
    print(f"   • Medical Imaging: {analysis['average_confidence']['Medical Imaging']:.3f}")
    print(f"   • Industrial Inspection: {analysis['average_confidence']['Industrial Inspection']:.3f}")
    print(f"   • Range: {analysis['confidence_range']['min']:.2f} - {analysis['confidence_range']['max']:.2f}")
    
    print(f"\n🎯 Key Observations:")
    for i, obs in enumerate(analysis['key_observations'], 1):
        print(f"   {i}. {obs}")
    
    print("\n" + "="*70)
    
    # Detailed breakdown
    print(f"\n📋 Detailed Breakdown by Example:")
    print("-" * 70)
    
    for idx, ex in enumerate(examples, 1):
        print(f"\n{idx}. {ex['prediction']} ({ex['domain']})")
        print(f"   Model: {ex['model_used']}")
        print(f"   Confidence: {ex['confidence']:.2%}")
        print(f"   Top-3 Predictions:")
        for i, pred in enumerate(ex['top_predictions'], 1):
            print(f"      {i}. {pred['label']}: {pred['score']:.2%}")
    
    print("\n" + "="*70)


def main():
    """Main visualization pipeline"""
    print("\n🚀 Starting Reference Predictions Visualization...")
    print("="*70)
    
    # Load data
    data = load_reference_predictions()
    print(f"✓ Loaded {len(data['examples'])} examples")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    plot_confidence_distribution(data)
    plot_top_predictions_scores(data)
    plot_model_comparison(data)
    plot_prediction_categories(data)
    
    # Generate summary
    generate_summary_stats(data)
    
    print("\n✅ All visualizations generated successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("   • confidence_analysis.png")
    print("   • top_predictions_breakdown.png")
    print("   • model_comparison.png")
    print("   • prediction_categories.png")
    print("\n")


if __name__ == "__main__":
    main()
