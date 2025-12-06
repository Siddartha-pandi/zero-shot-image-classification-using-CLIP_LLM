"""
Domain adaptation module for adaptive auto-tuning
"""

import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class DomainAdaptation:
    def __init__(self):
        # Domain-specific weight adjustments with feature emphasis
        self.domain_weights = {
            'natural_image': {'base': 1.0, 'adjustment': 0.0, 'color': 0.8, 'texture': 0.7, 'edges': 0.6, 'shape': 0.8},
            'sketch': {'base': 0.8, 'adjustment': 0.2, 'color': 0.05, 'texture': 0.2, 'edges': 1.0, 'shape': 1.0},
            'artistic_image': {'base': 0.9, 'adjustment': 0.1, 'color': 0.7, 'texture': 0.4, 'edges': 0.8, 'shape': 0.9},
            'medical_image': {'base': 1.1, 'adjustment': -0.1, 'color': 0.2, 'texture': 0.9, 'edges': 0.8, 'shape': 0.9},
            'multispectral_image': {'base': 1.2, 'adjustment': -0.2, 'color': 0.3, 'texture': 0.8, 'edges': 0.5, 'shape': 0.6},
            'modern_technology': {'base': 0.95, 'adjustment': 0.05, 'color': 0.7, 'texture': 0.6, 'edges': 0.7, 'shape': 0.85},
            'anime': {'base': 0.88, 'adjustment': 0.12, 'color': 0.75, 'texture': 0.3, 'edges': 0.85, 'shape': 0.9},
            'unknown': {'base': 1.0, 'adjustment': 0.0, 'color': 0.5, 'texture': 0.5, 'edges': 0.5, 'shape': 0.5}
        }

    
    def detect_domain(self, image_embeddings: np.ndarray) -> Dict[str, Any]:
        """Detect domain of the image based on embeddings with enhanced heuristics"""
        try:
            # Enhanced domain detection based on embedding patterns
            embedding_std = np.std(image_embeddings)
            embedding_mean = np.mean(image_embeddings)
            embedding_max = np.max(image_embeddings)
            embedding_min = np.min(image_embeddings)
            embedding_range = embedding_max - embedding_min
            
            # Advanced heuristic-based domain classification
            domain_scores = {}
            
            # Medical images: low variance, specific range patterns
            if embedding_std < 0.12 and embedding_range < 0.5:
                domain_scores['medical_image'] = 0.85
            
            # Sketches: high variance, edge-focused
            if embedding_std > 0.28:
                domain_scores['sketch'] = 0.75
            
            # Natural images: balanced statistics
            if 0.12 <= embedding_std <= 0.28 and abs(embedding_mean) < 0.08:
                domain_scores['natural_image'] = 0.90
            
            # Multispectral/Satellite: specific mean patterns
            if embedding_mean > 0.12 or embedding_range > 0.6:
                domain_scores['multispectral_image'] = 0.80
            
            # Artistic/Anime: moderate variance with specific patterns
            if 0.15 <= embedding_std <= 0.25 and embedding_mean < 0:
                domain_scores['anime'] = 0.85
                domain_scores['artistic_image'] = 0.82
            
            # Modern technology: specific patterns
            if 0.18 <= embedding_std <= 0.30 and 0.05 <= abs(embedding_mean) <= 0.15:
                domain_scores['modern_technology'] = 0.78
            
            # Select domain with highest score
            if domain_scores:
                domain = max(domain_scores, key=domain_scores.get)
                confidence = domain_scores[domain]
            else:
                domain = 'natural_image'
                confidence = 0.60
            
            # Determine characteristics
            characteristics = self._get_domain_characteristics(domain, embedding_std, embedding_mean)
            
            return {
                'domain': domain,
                'confidence': float(confidence),
                'characteristics': characteristics,
                'embedding_stats': {
                    'mean': float(embedding_mean),
                    'std': float(embedding_std),
                    'range': float(embedding_range)
                }
            }
            
        except Exception as e:
            logger.error(f"Error detecting domain: {str(e)}")
            return {
                'domain': 'unknown',
                'confidence': 0.5,
                'embedding_stats': {}
            }
    
    def adaptive_auto_tuning(self, similarity_scores: List[float], domain: str, domain_confidence: float = 1.0) -> List[float]:
        """Apply domain-specific adaptive tuning to similarity scores"""
        try:
            if domain not in self.domain_weights:
                domain = 'unknown'
            
            weights = self.domain_weights[domain]
            base_weight = weights['base']
            adjustment = weights['adjustment']
            
            # Apply domain-specific adjustments
            tuned_scores = []
            for score in similarity_scores:
                # Apply base weight and adjustment based on domain confidence
                adjusted_score = score * base_weight + (adjustment * domain_confidence)
                tuned_scores.append(adjusted_score)
            
            logger.info(f"Applied domain adaptation for '{domain}' domain (confidence: {domain_confidence:.3f})")
            
            return tuned_scores
            
        except Exception as e:
            logger.error(f"Error in adaptive auto-tuning: {str(e)}")
            return similarity_scores
    
    def _get_domain_characteristics(self, domain: str, std: float, mean: float) -> List[str]:
        """Get characteristics based on domain and embedding statistics"""
        characteristics = []
        
        if domain == 'sketch':
            characteristics = ['monochrome', 'line_based', 'low_texture', 'high_edge_density']
        elif domain == 'medical_image':
            characteristics = ['grayscale', 'specialized_patterns', 'clinical_features', 'high_precision_required']
        elif domain == 'natural_image':
            characteristics = ['photorealistic', 'natural_lighting', 'rich_texture', 'color_diverse']
        elif domain == 'artistic_image' or domain == 'anime':
            characteristics = ['stylized', 'non_photorealistic', 'high_contrast', 'graphic_design']
        elif domain == 'multispectral_image':
            characteristics = ['satellite_imagery', 'multi_band_data', 'vegetation_analysis', 'remote_sensing']
        elif domain == 'modern_technology':
            characteristics = ['contemporary_objects', 'manufactured_items', 'geometric_shapes']
        
        return characteristics
    
    def get_domain_info(self, domain: str) -> Dict[str, Any]:
        """Get comprehensive information about a specific domain"""
        domain_descriptions = {
            'natural_image': 'Natural photographs with realistic lighting and textures',
            'sketch': 'Hand-drawn sketches with line art and minimal detail',
            'artistic_image': 'Stylized artistic images with creative visual interpretation',
            'anime': 'Japanese animation style with distinctive visual characteristics',
            'medical_image': 'Medical imaging data (X-ray, MRI, CT) with specialized visual patterns',
            'multispectral_image': 'Satellite or multispectral imagery with geographical/vegetation features',
            'modern_technology': 'Contemporary technological devices and gadgets',
            'unknown': 'Domain could not be determined reliably'
        }
        
        adaptation_strategies = {
            'natural_image': 'Balanced feature extraction with emphasis on color and texture',
            'sketch': 'Edge and shape-focused analysis with minimal color dependency',
            'artistic_image': 'Style-aware processing with tolerance for exaggerated features',
            'anime': 'Recognition of stylized proportions and symbolic visual elements',
            'medical_image': 'Clinical feature extraction with pathology detection capabilities',
            'multispectral_image': 'Spectral band analysis with vegetation index calculations',
            'modern_technology': 'Geometric pattern recognition for manufactured objects'
        }
        
        return {
            'domain': domain,
            'description': domain_descriptions.get(domain, 'Unknown domain'),
            'weights': self.domain_weights.get(domain, self.domain_weights['unknown']),
            'adaptation_strategy': adaptation_strategies.get(domain, f'Standard processing for {domain}')
        }
