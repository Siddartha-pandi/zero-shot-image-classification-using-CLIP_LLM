# Comprehensive Evaluation System
# Tests: 1) CLIP+prompts, 2) Adaptation, 3) LLM reasoning

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from PIL import Image

# Import backend services
from app.clip_service import (
    classify_image, 
    create_class_prototype, 
    CLASS_PROTOTYPES,
    CONFIDENCE_THRESHOLD
)
from app.domain_service import infer_domain_from_hint
from app.caption_service import generate_caption
from app.llm_service import llm_reason_and_label, llm_narrative


class EvaluationDataset:
    """Load and manage evaluation datasets in JSON format."""
    
    def __init__(self, dataset_path: str):
        self.path = Path(dataset_path)
        self.samples = self._load_dataset()
        
    def _load_dataset(self) -> List[Dict]:
        """
        Load dataset from JSON file.
        Expected format:
        [
            {
                "path": "path/to/image.jpg",
                "label": "cat",
                "domain": "natural"
            },
            ...
        ]
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        
        with open(self.path, 'r') as f:
            data = json.load(f)
        
        # Validate format
        for i, sample in enumerate(data):
            required = ['path', 'label', 'domain']
            for field in required:
                if field not in sample:
                    raise ValueError(f"Sample {i} missing required field: {field}")
        
        return data
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def get_labels(self) -> List[str]:
        """Get unique labels in dataset."""
        return sorted(list(set(s['label'].lower() for s in self.samples)))
    
    def get_domains(self) -> List[str]:
        """Get unique domains in dataset."""
        return sorted(list(set(s['domain'] for s in self.samples)))
    
    def filter_by_domain(self, domain: str) -> List[Dict]:
        """Get samples from specific domain."""
        return [s for s in self.samples if s['domain'] == domain]
    
    def filter_by_labels(self, labels: List[str]) -> List[Dict]:
        """Get samples with specific labels."""
        label_set = set(l.lower() for l in labels)
        return [s for s in self.samples if s['label'].lower() in label_set]


class EvaluationMetrics:
    """Track and compute evaluation metrics."""
    
    def __init__(self):
        self.predictions = []
        self.ground_truths = []
        self.confidences = []
        self.domains = []
        self.latencies = []
        
        # Per-domain tracking
        self.domain_correct = defaultdict(int)
        self.domain_total = defaultdict(int)
        
        # Per-class tracking
        self.class_correct = defaultdict(int)
        self.class_total = defaultdict(int)
        
        # Top-k tracking
        self.top5_candidates = []
        
    def add_prediction(self, pred_label: str, true_label: str, 
                      confidence: float, domain: str, 
                      top5: List[str], latency: float = 0.0):
        """Add a single prediction result."""
        self.predictions.append(pred_label.lower())
        self.ground_truths.append(true_label.lower())
        self.confidences.append(confidence)
        self.domains.append(domain)
        self.top5_candidates.append([c.lower() for c in top5])
        self.latencies.append(latency)
        
        # Update per-domain
        self.domain_total[domain] += 1
        if pred_label.lower() == true_label.lower():
            self.domain_correct[domain] += 1
        
        # Update per-class
        self.class_total[true_label.lower()] += 1
        if pred_label.lower() == true_label.lower():
            self.class_correct[true_label.lower()] += 1
    
    def compute_metrics(self) -> Dict:
        """Compute all metrics."""
        if not self.predictions:
            return {}
        
        n = len(self.predictions)
        
        # Top-1 accuracy
        top1_correct = sum(1 for p, t in zip(self.predictions, self.ground_truths) if p == t)
        top1_acc = top1_correct / n
        
        # Top-5 accuracy
        top5_correct = sum(1 for t, candidates in zip(self.ground_truths, self.top5_candidates) 
                          if t in candidates)
        top5_acc = top5_correct / n
        
        # Per-domain accuracy
        domain_acc = {
            domain: self.domain_correct[domain] / self.domain_total[domain]
            for domain in self.domain_total.keys()
        }
        
        # Per-class accuracy
        class_acc = {
            cls: self.class_correct[cls] / self.class_total[cls]
            for cls in self.class_total.keys()
        }
        
        # Average latency
        avg_latency = np.mean(self.latencies) if self.latencies else 0.0
        
        # Confidence calibration (ECE)
        ece = self._compute_ece()
        
        return {
            'top1_accuracy': top1_acc,
            'top5_accuracy': top5_acc,
            'num_samples': n,
            'domain_accuracy': domain_acc,
            'class_accuracy': class_acc,
            'avg_latency_ms': avg_latency * 1000,
            'expected_calibration_error': ece,
            'avg_confidence': np.mean(self.confidences) if self.confidences else 0.0
        }
    
    def _compute_ece(self, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        if not self.confidences:
            return 0.0
        
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_mask = (np.array(self.confidences) >= bins[i]) & (np.array(self.confidences) < bins[i+1])
            if bin_mask.sum() == 0:
                continue
            
            bin_accuracy = np.mean([p == t for p, t, m in 
                                   zip(self.predictions, self.ground_truths, bin_mask) if m])
            bin_confidence = np.mean([c for c, m in zip(self.confidences, bin_mask) if m])
            bin_weight = bin_mask.sum() / len(self.confidences)
            
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def get_confusion_matrix(self) -> Dict:
        """Build confusion matrix."""
        labels = sorted(set(self.ground_truths))
        matrix = defaultdict(lambda: defaultdict(int))
        
        for pred, true in zip(self.predictions, self.ground_truths):
            matrix[true][pred] += 1
        
        return {
            'labels': labels,
            'matrix': {t: dict(matrix[t]) for t in labels}
        }


class ComprehensiveEvaluator:
    """Main evaluation framework with ablation study."""
    
    def __init__(self, dataset: EvaluationDataset):
        self.dataset = dataset
        self.results = {}
        
    def initialize_classes(self, labels: Optional[List[str]] = None):
        """Initialize class prototypes from text only (zero-shot)."""
        if labels is None:
            labels = self.dataset.get_labels()
        
        print(f"Initializing {len(labels)} classes...")
        for label in labels:
            # Infer domain from label (or use 'natural' as default)
            domain = 'natural'  # Could be smarter
            create_class_prototype(label=label, domain=domain, images=None)
        
        print(f"✓ {len(CLASS_PROTOTYPES)} class prototypes created")
    
    def run_baseline_clip(self, adapt: bool = False) -> Dict:
        """
        Baseline CLIP evaluation.
        - Single simple prompt per class
        - No domain-aware prompts
        - Optional: adaptation if adapt=True
        """
        print(f"\n{'='*60}")
        print(f"BASELINE CLIP (adapt={adapt})")
        print(f"{'='*60}")
        
        metrics = EvaluationMetrics()
        
        for i, sample in enumerate(self.dataset.samples):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(self.dataset)} images")
            
            try:
                img = Image.open(sample['path']).convert('RGB')
                true_label = sample['label'].lower()
                domain = sample['domain']
                
                start = time.time()
                result = classify_image(img, top_k=5)
                latency = time.time() - start
                
                pred_label = result['candidates'][0]['label'] if result['candidates'] else 'unknown'
                confidence = result['candidates'][0]['confidence'] if result['candidates'] else 0.0
                top5 = [c['label'] for c in result['candidates'][:5]]
                
                metrics.add_prediction(pred_label, true_label, confidence, domain, top5, latency)
                
            except Exception as e:
                print(f"Error on {sample['path']}: {e}")
                continue
        
        return metrics.compute_metrics()
    
    def run_with_domain_prompts(self) -> Dict:
        """
        Evaluation with domain-aware prompt engineering.
        Uses build_prompts_for_label with correct domain detection.
        """
        print(f"\n{'='*60}")
        print(f"WITH DOMAIN-AWARE PROMPTS")
        print(f"{'='*60}")
        
        metrics = EvaluationMetrics()
        
        for i, sample in enumerate(self.dataset.samples):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(self.dataset)} images")
            
            try:
                img = Image.open(sample['path']).convert('RGB')
                true_label = sample['label'].lower()
                domain = sample['domain']
                
                # Domain is already provided in dataset
                start = time.time()
                result = classify_image(img, top_k=5)
                latency = time.time() - start
                
                pred_label = result['candidates'][0]['label'] if result['candidates'] else 'unknown'
                confidence = result['candidates'][0]['confidence'] if result['candidates'] else 0.0
                top5 = [c['label'] for c in result['candidates'][:5]]
                
                metrics.add_prediction(pred_label, true_label, confidence, domain, top5, latency)
                
            except Exception as e:
                print(f"Error on {sample['path']}: {e}")
                continue
        
        return metrics.compute_metrics()
    
    def run_with_adaptation_tracking(self) -> Tuple[Dict, List[float]]:
        """
        Track accuracy over time as prototypes adapt.
        Returns final metrics + accuracy curve.
        """
        print(f"\n{'='*60}")
        print(f"WITH AUTO-TUNING (tracking over time)")
        print(f"{'='*60}")
        
        metrics = EvaluationMetrics()
        accuracy_curve = []
        window_size = 50
        
        for i, sample in enumerate(self.dataset.samples):
            try:
                img = Image.open(sample['path']).convert('RGB')
                true_label = sample['label'].lower()
                domain = sample['domain']
                
                start = time.time()
                result = classify_image(img, top_k=5)  # Adaptation happens inside
                latency = time.time() - start
                
                pred_label = result['candidates'][0]['label'] if result['candidates'] else 'unknown'
                confidence = result['candidates'][0]['confidence'] if result['candidates'] else 0.0
                top5 = [c['label'] for c in result['candidates'][:5]]
                
                metrics.add_prediction(pred_label, true_label, confidence, domain, top5, latency)
                
                # Track accuracy in windows
                if (i + 1) % window_size == 0:
                    current_acc = sum(1 for p, t in zip(metrics.predictions[-window_size:], 
                                                        metrics.ground_truths[-window_size:]) 
                                     if p == t) / window_size
                    accuracy_curve.append(current_acc)
                    print(f"Step {i+1}: Window accuracy = {current_acc:.3f}")
                
            except Exception as e:
                print(f"Error on {sample['path']}: {e}")
                continue
        
        final_metrics = metrics.compute_metrics()
        return final_metrics, accuracy_curve
    
    def run_online_learning_experiment(self, initial_classes: int = 10) -> Dict:
        """
        Test online class addition.
        Start with subset of classes, then add new ones dynamically.
        """
        print(f"\n{'='*60}")
        print(f"ONLINE LEARNING EXPERIMENT")
        print(f"{'='*60}")
        
        all_labels = self.dataset.get_labels()
        
        if len(all_labels) < initial_classes + 5:
            print(f"Warning: Not enough classes for online learning test")
            return {}
        
        # Split into initial and new classes
        initial_labels = all_labels[:initial_classes]
        new_labels = all_labels[initial_classes:initial_classes+10]
        
        print(f"Initial classes: {initial_labels}")
        print(f"New classes to add: {new_labels}")
        
        # Initialize only initial classes
        CLASS_PROTOTYPES.clear()
        self.initialize_classes(initial_labels)
        
        # Test on new class images BEFORE adding them
        new_class_samples = self.dataset.filter_by_labels(new_labels)
        metrics_before = EvaluationMetrics()
        
        print(f"\nEvaluating {len(new_class_samples)} new class samples BEFORE adding...")
        for sample in new_class_samples:
            try:
                img = Image.open(sample['path']).convert('RGB')
                result = classify_image(img, top_k=5)
                
                pred_label = result['candidates'][0]['label'] if result['candidates'] else 'unknown'
                confidence = result['candidates'][0]['confidence'] if result['candidates'] else 0.0
                top5 = [c['label'] for c in result['candidates'][:5]]
                
                metrics_before.add_prediction(pred_label, sample['label'].lower(), 
                                             confidence, sample['domain'], top5)
            except Exception as e:
                continue
        
        # Add new classes using 1-2 example images
        print(f"\nAdding {len(new_labels)} new classes...")
        for label in new_labels:
            # Get 1-2 example images for this class
            examples = [s for s in new_class_samples if s['label'].lower() == label.lower()][:2]
            example_images = [Image.open(ex['path']).convert('RGB') for ex in examples]
            
            create_class_prototype(label=label, domain='natural', images=example_images)
            print(f"  Added: {label} with {len(example_images)} examples")
        
        # Test again AFTER adding
        metrics_after = EvaluationMetrics()
        
        print(f"\nEvaluating {len(new_class_samples)} new class samples AFTER adding...")
        for sample in new_class_samples:
            try:
                img = Image.open(sample['path']).convert('RGB')
                result = classify_image(img, top_k=5)
                
                pred_label = result['candidates'][0]['label'] if result['candidates'] else 'unknown'
                confidence = result['candidates'][0]['confidence'] if result['candidates'] else 0.0
                top5 = [c['label'] for c in result['candidates'][:5]]
                
                metrics_after.add_prediction(pred_label, sample['label'].lower(), 
                                            confidence, sample['domain'], top5)
            except Exception as e:
                continue
        
        return {
            'before_addition': metrics_before.compute_metrics(),
            'after_addition': metrics_after.compute_metrics(),
            'improvement': metrics_after.compute_metrics()['top1_accuracy'] - 
                          metrics_before.compute_metrics()['top1_accuracy']
        }
    
    def run_with_llm(self) -> Dict:
        """
        Full system with LLM reasoning and re-ranking.
        Compare CLIP top-1 vs LLM final decision.
        """
        print(f"\n{'='*60}")
        print(f"FULL SYSTEM WITH LLM")
        print(f"{'='*60}")
        
        metrics_clip = EvaluationMetrics()
        metrics_llm = EvaluationMetrics()
        
        llm_improvements = []  # Cases where LLM fixed CLIP mistake
        
        for i, sample in enumerate(self.dataset.samples):
            if i % 20 == 0:  # LLM is slower
                print(f"Progress: {i}/{len(self.dataset)} images")
            
            try:
                img = Image.open(sample['path']).convert('RGB')
                true_label = sample['label'].lower()
                domain = sample['domain']
                
                # CLIP classification
                start = time.time()
                result = classify_image(img, top_k=5)
                clip_time = time.time() - start
                
                clip_label = result['candidates'][0]['label'] if result['candidates'] else 'unknown'
                confidence = result['candidates'][0]['confidence'] if result['candidates'] else 0.0
                top5 = [c['label'] for c in result['candidates'][:5]]
                
                # Add CLIP metrics
                metrics_clip.add_prediction(clip_label, true_label, confidence, domain, top5, clip_time)
                
                # LLM reasoning
                start = time.time()
                caption = generate_caption(img)
                reasoning = llm_reason_and_label(
                    caption=caption,
                    candidates=result['candidates'],
                    user_hint="",
                    domain=domain
                )
                llm_time = time.time() - start
                
                llm_label = reasoning['label'].lower()
                
                # Add LLM metrics (reuse CLIP confidence and top5)
                metrics_llm.add_prediction(llm_label, true_label, confidence, domain, top5, 
                                          clip_time + llm_time)
                
                # Track cases where LLM fixed CLIP error
                if clip_label != true_label and llm_label == true_label:
                    llm_improvements.append({
                        'image': sample['path'],
                        'true_label': true_label,
                        'clip_label': clip_label,
                        'llm_label': llm_label,
                        'reasoning': reasoning['reason']
                    })
                
            except Exception as e:
                print(f"Error on {sample['path']}: {e}")
                continue
        
        return {
            'clip_metrics': metrics_clip.compute_metrics(),
            'llm_metrics': metrics_llm.compute_metrics(),
            'llm_improvements': len(llm_improvements),
            'improvement_examples': llm_improvements[:5]  # First 5 examples
        }
    
    def run_full_ablation(self) -> Dict:
        """
        Run complete ablation study.
        Tests all 5 configurations.
        """
        results = {}
        
        # 1. Baseline CLIP (no adaptation)
        self.initialize_classes()
        results['1_baseline_clip'] = self.run_baseline_clip(adapt=False)
        
        # 2. With domain-aware prompts (already default in your system)
        results['2_domain_prompts'] = self.run_with_domain_prompts()
        
        # 3. With auto-tuning
        results['3_auto_tuning'], results['3_adaptation_curve'] = self.run_with_adaptation_tracking()
        
        # 4. Online learning
        results['4_online_learning'] = self.run_online_learning_experiment()
        
        # 5. Full system with LLM
        results['5_full_system_llm'] = self.run_with_llm()
        
        return results


def save_results(results: Dict, output_path: str):
    """Save evaluation results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Results saved to: {output_path}")


def print_summary_table(results: Dict):
    """Print ablation study summary table."""
    print(f"\n{'='*80}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*80}")
    print(f"{'Configuration':<40} {'Top-1':<10} {'Top-5':<10} {'Latency (ms)':<15}")
    print(f"{'-'*80}")
    
    configs = [
        ('1. CLIP only (baseline)', results.get('1_baseline_clip', {})),
        ('2. + Domain-aware prompts', results.get('2_domain_prompts', {})),
        ('3. + Auto-tuning', results.get('3_auto_tuning', {})),
        ('4. + Online learning', results.get('4_online_learning', {}).get('after_addition', {})),
        ('5. + LLM reasoning', results.get('5_full_system_llm', {}).get('llm_metrics', {}))
    ]
    
    for name, metrics in configs:
        if metrics:
            top1 = f"{metrics.get('top1_accuracy', 0)*100:.1f}%"
            top5 = f"{metrics.get('top5_accuracy', 0)*100:.1f}%"
            latency = f"{metrics.get('avg_latency_ms', 0):.1f}"
            print(f"{name:<40} {top1:<10} {top5:<10} {latency:<15}")
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive CLIP-LLM Evaluation')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to evaluation dataset JSON file')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output path for results')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'baseline', 'prompts', 'adaptation', 'online', 'llm'],
                       help='Evaluation mode')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from: {args.dataset}")
    dataset = EvaluationDataset(args.dataset)
    print(f"✓ Loaded {len(dataset)} samples")
    print(f"  Domains: {dataset.get_domains()}")
    print(f"  Classes: {len(dataset.get_labels())} unique labels")
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator(dataset)
    
    if args.mode == 'full':
        results = evaluator.run_full_ablation()
        print_summary_table(results)
    elif args.mode == 'baseline':
        evaluator.initialize_classes()
        results = {'baseline': evaluator.run_baseline_clip(adapt=False)}
    elif args.mode == 'prompts':
        evaluator.initialize_classes()
        results = {'prompts': evaluator.run_with_domain_prompts()}
    elif args.mode == 'adaptation':
        evaluator.initialize_classes()
        metrics, curve = evaluator.run_with_adaptation_tracking()
        results = {'adaptation': metrics, 'curve': curve}
    elif args.mode == 'online':
        results = {'online': evaluator.run_online_learning_experiment()}
    elif args.mode == 'llm':
        evaluator.initialize_classes()
        results = {'llm': evaluator.run_with_llm()}
    
    # Save results
    save_results(results, args.output)
