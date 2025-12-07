"""Main evaluation framework."""

import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.clip_service import (
    classify_image, 
    create_class_prototype, 
    CLASS_PROTOTYPES
)
from app.caption_service import generate_caption
from app.llm_service import llm_reason_and_label

from .dataset import EvaluationDataset
from .metrics import EvaluationMetrics


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
        CLASS_PROTOTYPES.clear()
        
        for label in labels:
            domain = 'natural'  # Default domain for text-only initialization
            create_class_prototype(label=label, domain=domain, images=None)
        
        print(f"✓ {len(CLASS_PROTOTYPES)} class prototypes created")
    
    def run_baseline_clip(self) -> Dict:
        """Baseline CLIP evaluation."""
        print(f"\n{'='*60}")
        print(f"BASELINE CLIP")
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
    
    def run_with_adaptation_tracking(self) -> Tuple[Dict, List[float]]:
        """Track accuracy over time as prototypes adapt."""
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
                result = classify_image(img, top_k=5)
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
        """Test online class addition."""
        print(f"\n{'='*60}")
        print(f"ONLINE LEARNING EXPERIMENT")
        print(f"{'='*60}")
        
        all_labels = self.dataset.get_labels()
        
        if len(all_labels) < initial_classes + 5:
            print(f"Warning: Not enough classes for online learning test")
            return {}
        
        # Split into initial and new classes
        initial_labels = all_labels[:initial_classes]
        new_labels = all_labels[initial_classes:min(initial_classes+10, len(all_labels))]
        
        print(f"Initial classes ({len(initial_labels)}): {initial_labels}")
        print(f"New classes ({len(new_labels)}): {new_labels}")
        
        # Initialize only initial classes
        self.initialize_classes(initial_labels)
        
        # Test on new class images BEFORE adding them
        new_class_samples = self.dataset.filter_by_labels(new_labels)
        metrics_before = EvaluationMetrics()
        
        print(f"\nEvaluating {len(new_class_samples)} new class samples BEFORE adding...")
        for sample in new_class_samples[:50]:  # Limit to 50 for speed
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
            examples = [s for s in new_class_samples if s['label'].lower() == label.lower()][:2]
            if examples:
                example_images = [Image.open(ex['path']).convert('RGB') for ex in examples]
                create_class_prototype(label=label, domain='natural', images=example_images)
                print(f"  Added: {label} with {len(example_images)} examples")
        
        # Test again AFTER adding
        metrics_after = EvaluationMetrics()
        
        print(f"\nEvaluating {len(new_class_samples)} new class samples AFTER adding...")
        for sample in new_class_samples[:50]:
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
        
        before_metrics = metrics_before.compute_metrics()
        after_metrics = metrics_after.compute_metrics()
        
        return {
            'before_addition': before_metrics,
            'after_addition': after_metrics,
            'improvement': after_metrics.get('top1_accuracy', 0) - before_metrics.get('top1_accuracy', 0)
        }
    
    def run_with_llm(self, max_samples: int = 100) -> Dict:
        """Full system with LLM reasoning."""
        print(f"\n{'='*60}")
        print(f"FULL SYSTEM WITH LLM (limited to {max_samples} samples)")
        print(f"{'='*60}")
        
        metrics_clip = EvaluationMetrics()
        metrics_llm = EvaluationMetrics()
        
        llm_improvements = []
        
        samples = self.dataset.samples[:max_samples]
        
        for i, sample in enumerate(samples):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(samples)} images")
            
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
                
                metrics_llm.add_prediction(llm_label, true_label, confidence, domain, top5, 
                                          clip_time + llm_time)
                
                # Track improvements
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
            'improvement_examples': llm_improvements[:5]
        }
    
    def run_full_ablation(self) -> Dict:
        """Run complete ablation study."""
        results = {}
        
        # 1. Baseline CLIP
        self.initialize_classes()
        results['1_baseline_clip'] = self.run_baseline_clip()
        
        # 2. With auto-tuning (reinitialize for clean start)
        self.initialize_classes()
        results['3_auto_tuning'], results['3_adaptation_curve'] = self.run_with_adaptation_tracking()
        
        # 3. Online learning
        results['4_online_learning'] = self.run_online_learning_experiment()
        
        # 4. Full system with LLM (limited samples due to speed)
        self.initialize_classes()
        results['5_full_system_llm'] = self.run_with_llm(max_samples=50)
        
        return results
