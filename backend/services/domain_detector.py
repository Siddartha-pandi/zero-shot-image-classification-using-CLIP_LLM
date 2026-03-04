# backend/services/domain_detector.py
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional
import logging
import base64
import io
import json

from config import DOMAINS, DOMAIN_PROMPTS
from models.clip_model import get_vith14_model
from models.llm_model import get_llm_model

logger = logging.getLogger(__name__)

class DomainDetector:
    def __init__(self):
        self.clip = get_vith14_model()
        self.llm = get_llm_model()

    def _get_llm_domain_prediction(self, image: Image.Image) -> Optional[Tuple[str, float]]:
        """
        Use LLM with vision to predict domain.
        Returns: (predicted_domain, confidence) or None if LLM not available
        """
        try:
            if self.llm.model is None:
                return None
            
            # Prepare image for Gemini
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Create prompt for domain detection
            domain_list = ", ".join(DOMAINS)
            prompt = f"""Analyze this image and determine which domain it belongs to.

Available domains: {domain_list}

Provide your answer in JSON format with exactly this structure:
{{
    "domain": "the most appropriate domain from the list",
    "confidence": 0.85,
    "reasoning": "brief explanation of why this domain was chosen"
}}

Be precise and choose only from the available domains."""

            # Generate with vision model
            from google.generativeai import types
            response = self.llm.model.generate_content(
                [prompt, {"mime_type": "image/png", "data": img_str}],
                generation_config=types.GenerationConfig(temperature=0.2, max_output_tokens=200)
            )
            
            # Parse JSON response
            result_text = response.text.strip()
            # Remove markdown code blocks if present
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()
            
            result = json.loads(result_text)
            predicted_domain = result.get("domain", "").lower()
            llm_confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "")
            
            # Validate predicted domain
            if predicted_domain in DOMAIN_PROMPTS:
                logger.info(f"LLM domain prediction: {predicted_domain} (confidence: {llm_confidence:.3f}) - {reasoning}")
                return predicted_domain, llm_confidence
            else:
                logger.warning(f"LLM predicted invalid domain: {predicted_domain}")
                return None
                
        except Exception as e:
            logger.warning(f"LLM domain prediction failed: {e}")
            return None

    def detect_domain(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        """
        Dynamically detects the domain of the image using both CLIP and LLM.
        Combines both predictions for better accuracy.
        Returns: (best_domain, confidence_score, all_domain_scores)
        """
        # === CLIP-based Detection ===
        image_emb = self.clip.encode_image(image)
        
        # Prepare configured domains
        domain_labels = list(DOMAIN_PROMPTS.keys())
        domain_texts = [DOMAIN_PROMPTS[d] for d in domain_labels]
        
        # Encode domains
        text_embs = self.clip.encode_text(domain_texts)
        
        # Compute similarity
        clip_similarities = self.clip.compute_similarity(image_emb, text_embs)
        
        # Normalize CLIP scores
        clip_scores = {
            domain_labels[i]: float(clip_similarities[i])
            for i in range(len(domain_labels))
        }
        
        # === LLM-based Detection ===
        llm_prediction = self._get_llm_domain_prediction(image)
        
        # === Combine Predictions ===
        if llm_prediction:
            llm_domain, llm_confidence = llm_prediction
            
            # Hybrid scoring: weighted combination
            # CLIP: 40%, LLM: 60% (LLM has better semantic understanding)
            combined_scores = {}
            for domain in domain_labels:
                clip_score = clip_scores[domain]
                # Boost LLM's predicted domain, dampen others
                llm_boost = llm_confidence if domain == llm_domain else (1 - llm_confidence) / (len(domain_labels) - 1)
                combined_scores[domain] = 0.4 * clip_score + 0.6 * llm_boost
            
            # Get best combined domain
            best_domain = max(combined_scores, key=combined_scores.get)
            confidence = combined_scores[best_domain]
            
            logger.info(f"Hybrid domain detection: {best_domain} (confidence: {confidence:.3f}, CLIP: {clip_scores[best_domain]:.3f}, LLM: {llm_domain})")
            return best_domain, confidence, combined_scores
        else:
            # Fallback to CLIP-only if LLM fails
            top_idx = np.argmax(clip_similarities)
            best_domain = domain_labels[top_idx]
            confidence = float(clip_similarities[top_idx])
            
            logger.info(f"CLIP-only domain detection: {best_domain} (confidence: {confidence:.3f})")
            return best_domain, confidence, clip_scores

_detector = None
def get_domain_detector() -> DomainDetector:
    global _detector
    if _detector is None:
        _detector = DomainDetector()
    return _detector
