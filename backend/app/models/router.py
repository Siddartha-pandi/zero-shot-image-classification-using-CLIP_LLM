# app/models/router.py
"""
Domain Router
Intelligently routes images to appropriate model (ViT-H/14 or MedCLIP)
"""
import numpy as np
from PIL import Image
from typing import Tuple, Literal
import logging

from .clip_vith14 import get_vith14_model
from .medclip_model import get_medclip_model

logger = logging.getLogger(__name__)

DomainType = Literal["medical", "fashion", "traffic", "satellite", "industrial", "natural"]

# Domain detection prompts - more distinctive keywords for better differentiation
DOMAIN_PROMPTS = {
    "medical": "x-ray scan radiograph ct mri ultrasound medical diagnosis hospital clinical anatomy organs tissue",
    "fashion": "clothing apparel fashion model outfit dress suit shoes accessories wear style",
    "traffic": "road street traffic car truck vehicle automobile pedestrian crossing intersection highway",
    "satellite": "aerial view satellite map birds eye top-down landscape grid pattern terrain",
    "industrial": "factory warehouse machinery equipment industrial production manufacturing heavy metal construction",
    "natural": "nature outdoor landscape scenery sky tree flower grass mountain water animal wildlife"
}

# Domain confidence thresholds - higher threshold for medical to avoid false positives
MEDICAL_THRESHOLD = 0.40  # Increased from 0.25 to be more conservative
TRAFFIC_THRESHOLD = 0.35  # Add minimum threshold for traffic detection

class DomainRouter:
    """Routes images to appropriate CLIP model based on domain"""
    
    def __init__(self):
        self.vith14 = get_vith14_model()
        self.medclip = get_medclip_model()
    
    def estimate_domain(self, image: Image.Image) -> Tuple[DomainType, float, dict]:
        """
        Estimate image domain using ViT-H/14
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (domain, confidence, all_scores)
        """
        # Encode image with ViT-H/14
        image_emb = self.vith14.encode_image(image)
        
        # Encode domain prompts
        domain_labels = list(DOMAIN_PROMPTS.keys())
        domain_texts = [DOMAIN_PROMPTS[d] for d in domain_labels]
        text_embs = self.vith14.encode_text(domain_texts)
        
        # Compute similarities
        similarities = self.vith14.compute_similarity(image_emb, text_embs)
        
        # Get top domain
        top_idx = np.argmax(similarities)
        domain = domain_labels[top_idx]
        confidence = float(similarities[top_idx])
        
        # Create scores dict
        all_scores = {
            domain_labels[i]: float(similarities[i])
            for i in range(len(domain_labels))
        }
        
        logger.info(f"Domain estimation: {domain} (confidence: {confidence:.3f})")
        
        return domain, confidence, all_scores  # type: ignore
    
    def route(self, image: Image.Image) -> Tuple[str, DomainType, float, dict]:
        """
        Route image to appropriate model
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (model_name, domain, confidence, domain_scores)
        """
        domain, confidence, domain_scores = self.estimate_domain(image)
        
        # Medical routing only if:
        # 1. Detected domain IS medical AND
        # 2. Medical score is high enough AND
        # 3. Medical score is noticeably better than natural score
        medical_score = domain_scores.get("medical", 0)
        natural_score = domain_scores.get("natural", 0)
        traffic_score = domain_scores.get("traffic", 0)
        
        # Use MedCLIP only if medical is clearly the best option
        if (domain == "medical" and 
            medical_score >= MEDICAL_THRESHOLD and 
            (medical_score - natural_score) > 0.05 and
            (medical_score - traffic_score) > 0.05):
            model_name = "MedCLIP"
            logger.info(f"Using MedCLIP: medical={medical_score:.3f}, natural={natural_score:.3f}, traffic={traffic_score:.3f}")
        else:
            # Use ViT-H/14 for all other cases (more robust for non-medical)
            model_name = "ViT-H/14"
            logger.info(f"Using ViT-H/14: domain={domain}, confidence={confidence:.3f}")
        
        logger.info(f"Routing to {model_name} for {domain} domain (score: {confidence:.3f})")
        
        return model_name, domain, confidence, domain_scores
    
    def classify_with_routing(
        self,
        image: Image.Image,
        labels: list,
        top_k: int = 5
    ) -> dict:
        """
        Classify image with automatic model routing
        
        Args:
            image: PIL Image
            labels: List of possible class labels
            top_k: Number of top predictions
            
        Returns:
            Classification results with routing information
        """
        # Determine which model to use
        model_name, domain, domain_conf, domain_scores = self.route(image)
        
        # Classify with appropriate model
        if model_name == "MedCLIP":
            predictions, image_emb = self.medclip.classify(image, labels, top_k)
        else:
            predictions, image_emb = self.vith14.classify(image, labels, top_k)
        
        return {
            "model_used": model_name,
            "domain": domain,
            "domain_confidence": domain_conf,
            "domain_scores": domain_scores,
            "predictions": predictions,
            "image_embedding": image_emb
        }


# Global instance
_router = None

def get_router() -> DomainRouter:
    """Get or create global router instance"""
    global _router
    if _router is None:
        _router = DomainRouter()
    return _router
