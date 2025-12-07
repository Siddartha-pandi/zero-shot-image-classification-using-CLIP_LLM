# Qualitative Evaluation Templates
# For human evaluation of LLM reasoning and narratives

import json
from pathlib import Path
from typing import List, Dict
import random


def generate_human_eval_template(
    evaluation_results_path: str,
    output_path: str = 'human_evaluation_template.json',
    num_samples: int = 50
):
    """
    Generate template for human evaluation of LLM outputs.
    Randomly selects samples from evaluation results.
    """
    
    # Load evaluation results
    with open(evaluation_results_path, 'r') as f:
        results = json.load(f)
    
    # Extract LLM examples (if available)
    llm_results = results.get('5_full_system_llm', {})
    improvements = llm_results.get('improvement_examples', [])
    
    # Create evaluation template
    template = {
        'instructions': {
            'overview': 'Please evaluate the quality of LLM reasoning and narratives',
            'time_estimate': f'{num_samples * 2} minutes (approximately 2 min per sample)',
            'rating_scales': {
                'label_correctness': {
                    '0': 'Completely wrong',
                    '1': 'Correct'
                },
                'reasoning_quality': {
                    '1': 'Wrong or nonsensical',
                    '2': 'Partially correct but vague',
                    '3': 'Generic but reasonable',
                    '4': 'Good, mentions relevant cues',
                    '5': 'Excellent, clear and specific visual cues'
                },
                'narrative_fluency': {
                    '1': 'Incoherent or ungrammatical',
                    '2': 'Basic but understandable',
                    '3': 'Clear and grammatical',
                    '4': 'Well-written and engaging',
                    '5': 'Excellent prose, natural flow'
                },
                'narrative_detail': {
                    '1': 'Very generic, no specifics',
                    '2': 'Some details but minimal',
                    '3': 'Moderate detail level',
                    '4': 'Good descriptive detail',
                    '5': 'Rich, vivid description'
                },
                'narrative_faithfulness': {
                    '1': 'Clear hallucinations or errors',
                    '2': 'Some questionable claims',
                    '3': 'Mostly faithful to image',
                    '4': 'Faithful with minor liberties',
                    '5': 'Completely faithful, no hallucinations'
                }
            }
        },
        'samples': [],
        'summary': {
            'total_samples': 0,
            'domains_covered': [],
            'evaluator_name': '',
            'date_completed': ''
        }
    }
    
    # Add samples for evaluation
    samples_for_eval = []
    
    # Prioritize LLM improvement cases
    for example in improvements[:min(20, len(improvements))]:
        samples_for_eval.append({
            'image_path': example['image'],
            'true_label': example['true_label'],
            'clip_prediction': example['clip_label'],
            'llm_prediction': example['llm_label'],
            'llm_reasoning': example.get('reasoning', ''),
            'category': 'llm_improvement'
        })
    
    # TODO: Add random samples from full dataset
    
    # Create evaluation entries
    for i, sample in enumerate(samples_for_eval[:num_samples]):
        template['samples'].append({
            'sample_id': i + 1,
            'image_path': sample['image_path'],
            'true_label': sample['true_label'],
            'clip_prediction': sample.get('clip_prediction', ''),
            'llm_prediction': sample['llm_prediction'],
            'llm_reasoning': sample.get('llm_reasoning', ''),
            'llm_narrative': sample.get('llm_narrative', '[Will be added during evaluation]'),
            
            # Fields for evaluator to fill
            'ratings': {
                'label_correctness': None,
                'reasoning_quality': None,
                'reasoning_mentions_correct_cues': None,  # boolean
                'narrative_fluency': None,
                'narrative_detail': None,
                'narrative_faithfulness': None
            },
            'comments': ''
        })
    
    template['summary']['total_samples'] = len(template['samples'])
    
    # Save template
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"✓ Human evaluation template saved to: {output_path}")
    print(f"  {len(template['samples'])} samples ready for evaluation")
    
    return template


def analyze_human_eval_results(eval_results_path: str) -> Dict:
    """
    Analyze completed human evaluation results.
    Compute average ratings and statistics.
    """
    
    with open(eval_results_path, 'r') as f:
        results = json.load(f)
    
    samples = results['samples']
    
    # Aggregate ratings
    label_correct_count = 0
    reasoning_scores = []
    narrative_fluency = []
    narrative_detail = []
    narrative_faithfulness = []
    reasoning_mentions_cues = 0
    
    for sample in samples:
        ratings = sample.get('ratings', {})
        
        if ratings.get('label_correctness') is not None:
            label_correct_count += ratings['label_correctness']
        
        if ratings.get('reasoning_quality') is not None:
            reasoning_scores.append(ratings['reasoning_quality'])
        
        if ratings.get('narrative_fluency') is not None:
            narrative_fluency.append(ratings['narrative_fluency'])
        
        if ratings.get('narrative_detail') is not None:
            narrative_detail.append(ratings['narrative_detail'])
        
        if ratings.get('narrative_faithfulness') is not None:
            narrative_faithfulness.append(ratings['narrative_faithfulness'])
        
        if ratings.get('reasoning_mentions_correct_cues'):
            reasoning_mentions_cues += 1
    
    n_samples = len(samples)
    
    analysis = {
        'num_samples_evaluated': n_samples,
        'evaluator': results['summary'].get('evaluator_name', 'Unknown'),
        'date': results['summary'].get('date_completed', 'Unknown'),
        
        'label_accuracy': label_correct_count / n_samples if n_samples > 0 else 0,
        
        'reasoning_quality': {
            'mean': sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0,
            'min': min(reasoning_scores) if reasoning_scores else 0,
            'max': max(reasoning_scores) if reasoning_scores else 0,
            'distribution': _count_distribution(reasoning_scores, 5)
        },
        
        'reasoning_mentions_correct_cues_pct': 
            reasoning_mentions_cues / n_samples if n_samples > 0 else 0,
        
        'narrative_fluency': {
            'mean': sum(narrative_fluency) / len(narrative_fluency) if narrative_fluency else 0,
            'distribution': _count_distribution(narrative_fluency, 5)
        },
        
        'narrative_detail': {
            'mean': sum(narrative_detail) / len(narrative_detail) if narrative_detail else 0,
            'distribution': _count_distribution(narrative_detail, 5)
        },
        
        'narrative_faithfulness': {
            'mean': sum(narrative_faithfulness) / len(narrative_faithfulness) if narrative_faithfulness else 0,
            'distribution': _count_distribution(narrative_faithfulness, 5)
        }
    }
    
    return analysis


def _count_distribution(values: List[int], max_val: int) -> Dict[int, int]:
    """Count distribution of rating values."""
    dist = {i: 0 for i in range(1, max_val + 1)}
    for v in values:
        if v in dist:
            dist[v] += 1
    return dist


def print_human_eval_summary(analysis: Dict):
    """Print formatted summary of human evaluation results."""
    
    print(f"\n{'='*60}")
    print("HUMAN EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Evaluator: {analysis['evaluator']}")
    print(f"Date: {analysis['date']}")
    print(f"Samples: {analysis['num_samples_evaluated']}")
    print(f"{'-'*60}")
    
    print(f"\nLABEL CORRECTNESS")
    print(f"  Accuracy: {analysis['label_accuracy']*100:.1f}%")
    
    print(f"\nREASONING QUALITY")
    print(f"  Mean score: {analysis['reasoning_quality']['mean']:.2f} / 5")
    print(f"  Range: {analysis['reasoning_quality']['min']} - {analysis['reasoning_quality']['max']}")
    print(f"  Mentions correct cues: {analysis['reasoning_mentions_correct_cues_pct']*100:.1f}%")
    print(f"  Distribution: {analysis['reasoning_quality']['distribution']}")
    
    print(f"\nNARRATIVE QUALITY")
    print(f"  Fluency: {analysis['narrative_fluency']['mean']:.2f} / 5")
    print(f"  Detail: {analysis['narrative_detail']['mean']:.2f} / 5")
    print(f"  Faithfulness: {analysis['narrative_faithfulness']['mean']:.2f} / 5")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Qualitative Evaluation Tools')
    parser.add_argument('--action', choices=['generate', 'analyze'], required=True,
                       help='Generate template or analyze results')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path')
    parser.add_argument('--samples', type=int, default=50,
                       help='Number of samples for human evaluation')
    
    args = parser.parse_args()
    
    if args.action == 'generate':
        output = args.output or 'human_evaluation_template.json'
        generate_human_eval_template(args.input, output, args.samples)
    
    elif args.action == 'analyze':
        analysis = analyze_human_eval_results(args.input)
        print_human_eval_summary(analysis)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"✓ Analysis saved to: {args.output}")
