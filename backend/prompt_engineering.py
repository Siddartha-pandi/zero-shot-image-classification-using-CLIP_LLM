"""
Handles advanced prompt engineering strategies, including generation and
auto-tuning of prompt weights based on image-text similarity.
Supports multilingual prompts and domain-specific adaptations.
"""
import torch
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PromptEngineer:
    def __init__(self):
        # Comprehensive templates for style-agnostic adaptability
        self.base_templates = [
            "a photo of a {}.",
            "a photograph showing a {}.",
            "an image of a {}.",
        ]
        
        self.style_templates = [
            "a sketch of a {}.",
            "a drawing of a {}.",
            "a rendering of a {}.",
            "an illustration of a {}.",
            "artwork depicting a {}.",
        ]
        
        self.quality_templates = [
            "a high-quality photo of the {}.",
            "a low-quality photo of the {}.",
            "a blurry photo of the {}.",
            "a clear image of the {}.",
        ]
        
        self.viewpoint_templates = [
            "a cropped photo of the {}.",
            "a close-up photo of a {}.",
            "a photo of the small {}.",
            "a photo of the large {}.",
            "a distant view of the {}.",
        ]
        
        self.context_templates = [
            "a {} in the wild.",
            "a {} in its natural habitat.",
            "a {} in an indoor setting.",
            "a {} outdoors.",
        ]
        
        # Multilingual templates (English, Hindi, Spanish, French)
        self.multilingual_templates = {
            'en': [
                "a photo of a {}.",
                "an image showing a {}.",
            ],
            'hi': [
                "एक {} की तस्वीर।",
                "एक {} दिखाने वाली छवि।",
            ],
            'es': [
                "una foto de un {}.",
                "una imagen de un {}.",
            ],
            'fr': [
                "une photo d'un {}.",
                "une image d'un {}.",
            ]
        }
        
        # Domain-specific templates
        self.domain_templates = {
            'medical_image': [
                "a medical scan showing {}.",
                "a radiograph of {}.",
                "a clinical image of {}.",
                "a diagnostic image showing {}.",
            ],
            'sketch': [
                "a line drawing of a {}.",
                "a hand-drawn sketch of a {}.",
                "an outline drawing of a {}.",
            ],
            'anime': [
                "an anime-style illustration of a {}.",
                "a manga drawing of a {}.",
                "a stylized anime character showing {}.",
            ],
            'multispectral_image': [
                "satellite imagery showing {}.",
                "aerial view of {}.",
                "multispectral data indicating {}.",
            ]
        }

    def generate_prompts_for_class(self, class_name: str, domain: str = None, language: str = 'en') -> List[str]:
        """Generates a comprehensive list of prompts for a single class name."""
        prompts = []
        
        # Add base templates
        prompts.extend([template.format(class_name) for template in self.base_templates])
        
        # Add style templates
        prompts.extend([template.format(class_name) for template in self.style_templates[:3]])
        
        # Add quality templates
        prompts.extend([template.format(class_name) for template in self.quality_templates[:2]])
        
        # Add viewpoint templates
        prompts.extend([template.format(class_name) for template in self.viewpoint_templates[:2]])
        
        # Add domain-specific templates if domain is specified
        if domain and domain in self.domain_templates:
            domain_prompts = [template.format(class_name) for template in self.domain_templates[domain]]
            prompts.extend(domain_prompts)
            logger.info(f"Added {len(domain_prompts)} domain-specific prompts for {domain}")
        
        # Add multilingual templates if language is not English
        if language != 'en' and language in self.multilingual_templates:
            multilingual_prompts = [template.format(class_name) for template in self.multilingual_templates[language]]
            prompts.extend(multilingual_prompts)
            logger.info(f"Added multilingual prompts for language: {language}")
        
        return prompts
    
    def generate_semantic_expansion(self, class_name: str, domain: str = None) -> List[str]:
        """Generate semantic expansions for zero-shot reasoning."""
        expansions = [
            f"This appears to be a {class_name}.",
            f"The image shows a {class_name}.",
            f"Could be {class_name} or similar.",
            f"Looks like a {class_name}.",
        ]
        
        # Domain-specific semantic expansions
        if domain == 'medical_image':
            expansions.extend([
                f"Medical imaging showing {class_name}.",
                f"Clinical presentation of {class_name}.",
            ])
        elif domain == 'sketch':
            expansions.extend([
                f"A simplified drawing depicting {class_name}.",
                f"Line art representation of {class_name}.",
            ])
        elif domain == 'anime' or domain == 'artistic_image':
            expansions.extend([
                f"Stylized artwork showing {class_name}.",
                f"Artistic interpretation of {class_name}.",
            ])
        
        return expansions

    def auto_tune_weights(self, image_features, text_features_for_prompts):
        """
        Dynamically determines the best prompt template weights for a given image.
        Uses cosine similarity and softmax for weighted distribution.
        """
        with torch.no_grad():
            # Normalize features to ensure accurate similarity calculation
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_for_prompts = text_features_for_prompts / text_features_for_prompts.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarities
            similarities = (100.0 * image_features @ text_features_for_prompts.T)
            
            # Apply softmax to get weights
            weights = torch.nn.functional.softmax(similarities, dim=-1)
        return weights
