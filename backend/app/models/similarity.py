# app/models/similarity.py
"""
Similarity Computation Module
Handles confidence scoring with domain-specific formulas
"""
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    # Normalize vectors
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
    
    # Compute dot product
    similarity = np.dot(vec1_norm, vec2_norm)
    
    # Clamp to [0, 1]
    return float(np.clip(similarity, 0, 1))


def compute_confidence_score(
    domain: str,
    model_used: str,
    prediction_score: float,
    image_emb: np.ndarray,
    caption_emb: np.ndarray = None,
    domain_similarity: float = None
) -> float:
    """
    Compute final confidence score based on domain
    
    For medical images (MedCLIP):
        Confidence = 0.7 * Sim_disease + 0.3 * Sim_caption
        
    For non-medical (ViT-H/14):
        Confidence = cos(E_img, E_label)
    
    Args:
        domain: Image domain
        model_used: Model name (ViT-H/14 or MedCLIP)
        prediction_score: Score from model prediction
        image_emb: Image embedding
        caption_emb: Optional caption embedding
        domain_similarity: Optional domain similarity score
        
    Returns:
        Final confidence score (0-1)
    """
    if domain == "medical" and model_used == "MedCLIP":
        # Medical formula: 0.7 * disease_sim + 0.3 * caption_sim
        disease_sim = prediction_score
        
        if caption_emb is not None:
            caption_sim = compute_cosine_similarity(image_emb, caption_emb)
            confidence = 0.7 * disease_sim + 0.3 * caption_sim
        else:
            # Fallback if no caption available
            confidence = disease_sim
        
        logger.debug(f"Medical confidence: disease={disease_sim:.3f}, final={confidence:.3f}")
    else:
        # General formula: direct cosine similarity
        confidence = prediction_score
        
        logger.debug(f"General confidence: {confidence:.3f}")
    
    return float(np.clip(confidence, 0, 1))


def rank_predictions(
    predictions: List[Dict],
    image_emb: np.ndarray,
    model,
    top_k: int = 5
) -> List[Dict]:
    """
    Re-rank predictions with additional similarity computations
    
    Args:
        predictions: Initial predictions
        image_emb: Image embedding
        model: CLIP model instance
        top_k: Number of top results
        
    Returns:
        Re-ranked predictions with enhanced scores
    """
    enhanced_predictions = []
    
    for pred in predictions[:top_k]:
        label = pred["label"]
        base_score = pred["score"]
        
        # Compute text embedding for label
        text_emb = model.encode_text([label])[0]
        
        # Compute direct similarity
        direct_sim = compute_cosine_similarity(image_emb, text_emb)
        
        # Combine scores (weighted average)
        enhanced_score = 0.6 * base_score + 0.4 * direct_sim
        
        enhanced_predictions.append({
            "label": label,
            "score": float(enhanced_score),
            "base_score": float(base_score),
            "direct_similarity": float(direct_sim)
        })
    
    # Re-sort by enhanced score
    enhanced_predictions.sort(key=lambda x: x["score"], reverse=True)
    
    return enhanced_predictions


def compute_batch_similarities(
    image_embs: np.ndarray,
    text_embs: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise similarities for batch processing
    
    Args:
        image_embs: Image embeddings [N, D]
        text_embs: Text embeddings [M, D]
        
    Returns:
        Similarity matrix [N, M]
    """
    # Normalize embeddings
    image_embs_norm = image_embs / (np.linalg.norm(image_embs, axis=1, keepdims=True) + 1e-8)
    text_embs_norm = text_embs / (np.linalg.norm(text_embs, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity matrix
    similarities = image_embs_norm @ text_embs_norm.T
    
    return np.clip(similarities, 0, 1)


def get_confidence_explanation(
    confidence: float,
    domain: str,
    model_used: str
) -> str:
    """
    Generate human-readable confidence explanation
    
    Args:
        confidence: Confidence score
        domain: Image domain
        model_used: Model name
        
    Returns:
        Explanation string
    """
    if confidence >= 0.8:
        level = "Very High"
        desc = "strong visual match and clear features"
    elif confidence >= 0.6:
        level = "High"
        desc = "good visual match with recognizable features"
    elif confidence >= 0.4:
        level = "Moderate"
        desc = "reasonable visual match but some ambiguity"
    elif confidence >= 0.2:
        level = "Low"
        desc = "weak visual match with significant uncertainty"
    else:
        level = "Very Low"
        desc = "very weak visual match, highly uncertain"
    
    return f"{level} confidence ({confidence:.1%}) - {desc} detected by {model_used} in {domain} domain"
