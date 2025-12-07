"""Metrics tracking and computation."""

from typing import List, Dict
from collections import defaultdict
import numpy as np


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
            'top1_accuracy': float(top1_acc),
            'top5_accuracy': float(top5_acc),
            'num_samples': n,
            'domain_accuracy': domain_acc,
            'class_accuracy': class_acc,
            'avg_latency_ms': float(avg_latency * 1000),
            'expected_calibration_error': float(ece),
            'avg_confidence': float(np.mean(self.confidences)) if self.confidences else 0.0
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
    
    def get_confusion_pairs(self) -> List[tuple]:
        """Get confused prediction pairs."""
        confused = []
        for pred, true in zip(self.predictions, self.ground_truths):
            if pred != true:
                confused.append((true, pred))
        return confused
