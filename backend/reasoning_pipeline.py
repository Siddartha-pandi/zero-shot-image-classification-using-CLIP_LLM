"""
High-Level Reasoning Pipeline for Zero-Shot Image Classification

Pipeline stages:
1. CLIP Stage: Encode image → compute logits vs text labels
2. Domain Stage: Detect domain → apply adaptive module → adjust scores
3. Calibration Stage: Apply temperature scaling → probabilities
4. Caption + Features: Generate caption and extract attributes
5. LLM Re-ranking Stage: Re-order labels with adjusted probabilities
6. LLM Explanation Stage: Generate UI-friendly reasoning

Returns JSON shaped for the UI with clean structure.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import torch
import logging
import json

logger = logging.getLogger(__name__)


# ========== 1. Math Helpers ==========

def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Compute softmax with temperature scaling for calibrated probabilities.
    
    Args:
        logits: Raw similarity scores
        temperature: Scaling factor (lower = more confident, higher = more uniform)
    
    Returns:
        Calibrated probabilities that sum to 1
    """
    logits = np.array(logits, dtype=np.float64) / max(temperature, 1e-6)
    z = logits - logits.max()  # Numerical stability
    e = np.exp(z)
    return e / e.sum()


def topk(tensor: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get top-k indices and values from array.
    
    Returns:
        (indices, values) of top k elements
    """
    idx = np.argsort(tensor)[::-1][:k]
    return idx, tensor[idx]


# ========== 2. CLIP + Adaptive + Domain ==========

def encode_image(
    image: Image.Image,
    model,
    preprocess,
    adaptive_module=None,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Encode image using CLIP and optionally apply adaptive module.
    
    Args:
        image: PIL Image
        model: CLIP model
        preprocess: CLIP preprocessor
        adaptive_module: Optional domain-adaptive transformation
        device: Compute device
    
    Returns:
        Normalized image embedding (1, D)
    """
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        
        if adaptive_module is not None:
            emb = adaptive_module(emb)
            emb = emb / emb.norm(dim=-1, keepdim=True)
    
    return emb  # (1, D)


def detect_domain(image_emb: torch.Tensor) -> Dict[str, Any]:
    """
    Detect image domain from embedding statistics.
    
    Enhanced heuristic-based domain classification.
    Replace with trained classifier for production.
    
    Returns:
        Dict with domain, confidence, characteristics, embedding_stats
    """
    emb_np = image_emb.detach().cpu().numpy().flatten()
    
    embedding_std = float(np.std(emb_np))
    embedding_mean = float(np.mean(emb_np))
    embedding_max = float(np.max(emb_np))
    embedding_min = float(np.min(emb_np))
    embedding_range = embedding_max - embedding_min
    
    domain_scores = {}
    
    # Heuristic domain detection
    if embedding_std < 0.12 and embedding_range < 0.5:
        domain_scores['medical_image'] = 0.85
    
    if embedding_std > 0.28:
        domain_scores['sketch'] = 0.75
    
    if 0.12 <= embedding_std <= 0.28 and abs(embedding_mean) < 0.08:
        domain_scores['natural_image'] = 0.90
    
    if embedding_mean > 0.12 or embedding_range > 0.6:
        domain_scores['multispectral_image'] = 0.80
    
    if 0.15 <= embedding_std <= 0.25 and embedding_mean < 0:
        domain_scores['anime'] = 0.85
        domain_scores['artistic_image'] = 0.82
    
    if 0.18 <= embedding_std <= 0.30 and 0.05 <= abs(embedding_mean) <= 0.15:
        domain_scores['modern_technology'] = 0.78
    
    # Select best domain
    if domain_scores:
        domain = max(domain_scores, key=domain_scores.get)
        confidence = domain_scores[domain]
    else:
        domain = 'natural_image'
        confidence = 0.60
    
    # Domain characteristics
    characteristics_map = {
        'natural_image': ['photorealistic', 'natural_lighting', 'rich_texture'],
        'sketch': ['monochrome', 'line_based', 'low_texture'],
        'medical_image': ['grayscale', 'specialized_patterns', 'clinical_features'],
        'artistic_image': ['stylized', 'non_photorealistic', 'high_contrast'],
        'anime': ['stylized', 'exaggerated_features', 'graphic_design'],
        'multispectral_image': ['satellite_imagery', 'multi_band_data'],
        'modern_technology': ['contemporary_objects', 'manufactured_items']
    }
    
    return {
        "domain": domain,
        "confidence": float(confidence),
        "characteristics": characteristics_map.get(domain, []),
        "embedding_stats": {
            "mean": embedding_mean,
            "std": embedding_std,
            "range": embedding_range
        }
    }


def compute_clip_scores(
    image_emb: torch.Tensor,
    text_embs: torch.Tensor
) -> np.ndarray:
    """
    Compute CLIP similarity scores between image and text embeddings.
    
    Returns:
        Similarity scores array (num_labels,)
    """
    with torch.no_grad():
        logits = (image_emb @ text_embs.T).cpu().numpy().squeeze()
    return logits


# ========== 3. LLM Helpers ==========

def build_image_caption(
    llm_client,
    simple_tags: List[str],
    domain: str = "general"
) -> str:
    """
    Generate a short caption from image attributes using LLM.
    
    COPILOT: implement llm caption generation that takes tags and domain,
    produces a single natural sentence describing the image.
    
    Args:
        llm_client: LLM client with generate() method
        simple_tags: List of detected attributes/labels
        domain: Image domain context
    
    Returns:
        Short caption string
    """
    if llm_client is None:
        # Fallback: simple concatenation
        return f"Image showing {', '.join(simple_tags[:3])}"
    
    try:
        prompt = f"""Write one short caption (max 15 words) for a {domain} image with these attributes: {', '.join(simple_tags)}.
Output only the caption, no extra text."""
        
        caption = llm_client.generate(prompt).strip()
        # Remove quotes if LLM added them
        caption = caption.strip('"\'')
        return caption
    
    except Exception as e:
        logger.warning(f"LLM caption generation failed: {e}")
        return f"Image showing {', '.join(simple_tags[:3])}"


def llm_rerank(
    llm_client,
    caption: str,
    labels: List[str],
    probs: np.ndarray,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Use LLM to re-rank classification candidates based on semantic understanding.
    
    COPILOT: implement llm_rerank that sends caption + labels + probs to the LLM,
    gets back JSON with new label order and probabilities that sum to 1.
    
    Args:
        llm_client: LLM client with generate() method
        caption: Image caption/description
        labels: List of candidate labels
        probs: Probability for each label
        top_k: Number of candidates to re-rank
    
    Returns:
        List of (label, probability) tuples in new order
    """
    if llm_client is None:
        # Fallback: return original order
        idx, _ = topk(probs, min(top_k, len(labels)))
        return [(labels[i], float(probs[i])) for i in idx]
    
    try:
        cand_lines = "\n".join(
            f"{i+1}. {label} ({float(p):.4f})"
            for i, (label, p) in enumerate(zip(labels[:top_k], probs[:top_k]))
        )
        
        prompt = f"""You are re-ranking image classification labels for better accuracy.

Image caption: "{caption}"

Candidate labels with CLIP probabilities:
{cand_lines}

Task:
1. Re-rank the labels based on semantic match with the caption
2. Adjust probabilities if needed (they must sum to 1.0)
3. Return ONLY a JSON array, no other text

Format: [{{"label":"label1","prob":0.84}},{{"label":"label2","prob":0.12}}]

Output:"""
        
        raw = llm_client.generate(prompt).strip()
        
        # Extract JSON from response
        start = raw.find('[')
        end = raw.rfind(']') + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        
        parsed = json.loads(raw)
        
        # Normalize probabilities to sum to 1
        total = sum(item["prob"] for item in parsed)
        if total > 0:
            normalized = [(item["label"], float(item["prob"]) / total) for item in parsed]
        else:
            normalized = [(item["label"], 1.0 / len(parsed)) for item in parsed]
        
        logger.info(f"LLM re-ranking successful: {len(normalized)} labels")
        return normalized
    
    except Exception as e:
        logger.warning(f"LLM re-ranking failed: {e}, using original probabilities")
        # Fallback to original order
        idx, _ = topk(probs, min(top_k, len(labels)))
        return [(labels[i], float(probs[i])) for i in idx]


def generate_ui_reasoning(
    llm_client,
    top_label: str,
    caption: str,
    attributes: List[str],
    domain: str,
    confidence: float
) -> Dict[str, Any]:
    """
    Generate UI-friendly reasoning in the style of the Adaptive CLIP Output UI.
    
    COPILOT: implement generate_ui_reasoning returning JSON with summary,
    attributes list, and detailed_reasoning, in the style of the Adaptive CLIP Output UI.
    
    Args:
        llm_client: LLM client
        top_label: Predicted label
        caption: Image caption
        attributes: Extracted visual attributes
        domain: Detected domain
        confidence: Prediction confidence
    
    Returns:
        Dict with summary, attributes, detailed_reasoning
    """
    if llm_client is None:
        # Fallback: generate simple reasoning
        return {
            "summary": f"Classified as '{top_label}' with {confidence:.1%} confidence",
            "attributes": attributes[:5],
            "detailed_reasoning": f"The image shows {caption.lower()}. Based on visual analysis and domain detection ({domain}), it was classified as '{top_label}' with {confidence:.1%} confidence."
        }
    
    try:
        attr_str = ", ".join(attributes[:5])
        
        prompt = f"""You are an AI assistant explaining zero-shot image classification results.

Predicted Label: "{top_label}"
Confidence: {confidence:.1%}
Image Caption: "{caption}"
Domain: {domain}
Visual Attributes: {attr_str}

Generate a JSON object with:
1. "summary": One concise sentence (max 20 words) describing the classification result
2. "attributes": Array of 3-5 key factors that led to this classification (can polish the provided attributes)
3. "detailed_reasoning": 2-3 sentences explaining why this classification makes sense

Output ONLY valid JSON, no additional text.

Example format:
{{"summary": "...", "attributes": ["...", "..."], "detailed_reasoning": "..."}}

Output:"""
        
        raw = llm_client.generate(prompt).strip()
        
        # Extract JSON from response
        start = raw.find('{')
        end = raw.rfind('}') + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        
        reasoning = json.loads(raw)
        
        # Validate structure
        if not all(key in reasoning for key in ["summary", "attributes", "detailed_reasoning"]):
            raise ValueError("Missing required keys in LLM response")
        
        logger.info("LLM reasoning generation successful")
        return reasoning
    
    except Exception as e:
        logger.warning(f"LLM reasoning generation failed: {e}, using fallback")
        return {
            "summary": f"Classified as '{top_label}' with {confidence:.1%} confidence",
            "attributes": attributes[:5] if attributes else ["visual analysis", "domain detection", "semantic matching"],
            "detailed_reasoning": f"The image shows {caption.lower()}. Based on visual analysis in the {domain.replace('_', ' ')} domain, it was classified as '{top_label}' with {confidence:.1%} confidence using zero-shot CLIP embeddings and LLM enhancement."
        }


# ========== 4. Main Pipeline Function ==========

def run_reasoning_pipeline(
    image: Image.Image,
    label_names: List[str],
    text_embs: torch.Tensor,
    clip_model,
    clip_preprocess,
    adaptive_module=None,
    llm_client=None,
    temperature: float = 0.01,
    top_k: int = 5,
    use_llm_reranking: bool = True,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Full reasoning pipeline for zero-shot image classification.
    
    Pipeline:
    1. CLIP Stage: Encode image → compute logits
    2. Domain Stage: Detect domain → apply adaptive module
    3. Calibration: Temperature scaling → probabilities
    4. Caption: Generate image description
    5. LLM Re-ranking: Adjust probabilities semantically
    6. LLM Explanation: Generate UI-friendly reasoning
    
    Args:
        image: PIL Image to classify
        label_names: List of class labels
        text_embs: Pre-computed text embeddings for labels
        clip_model: CLIP model
        clip_preprocess: CLIP preprocessor
        adaptive_module: Optional domain-adaptive module
        llm_client: Optional LLM client for enhancement
        temperature: Softmax temperature (default 0.01)
        top_k: Number of top predictions to consider
        use_llm_reranking: Whether to use LLM re-ranking
        device: Compute device
    
    Returns:
        JSON-ready dict for UI with clean structure
    """
    logger.info(f"Running reasoning pipeline on image, {len(label_names)} labels")
    
    # 1) CLIP Stage: Encode image with adaptive module
    image_emb = encode_image(image, clip_model, clip_preprocess, adaptive_module, device)
    logger.info("Image encoded with CLIP")
    
    # 2) Domain Stage: Detect domain
    domain_info = detect_domain(image_emb)
    logger.info(f"Domain detected: {domain_info['domain']} ({domain_info['confidence']:.2%})")
    
    # 3) Calibration Stage: Raw CLIP scores → calibrated probabilities
    logits = compute_clip_scores(image_emb, text_embs)
    probs = softmax(logits, temperature=temperature)
    logger.info("Probabilities calibrated with temperature scaling")
    
    # 4) Get top-k candidates from CLIP
    idx, top_probs = topk(probs, min(top_k, len(label_names)))
    top_labels = [label_names[i] for i in idx]
    
    # 5) Caption + Features Stage: Extract attributes
    # Simple approach: use top labels as attributes
    # You can replace with a vision tagging model (BLIP, RAM, etc.)
    attributes = [label.replace('_', ' ').title() for label in top_labels[:4]]
    
    # Add domain characteristics
    if domain_info.get('characteristics'):
        attributes.extend([char.replace('_', ' ').title() for char in domain_info['characteristics'][:2]])
    
    # 6) Build caption
    caption = build_image_caption(llm_client, attributes, domain_info['domain'])
    logger.info(f"Caption generated: {caption}")
    
    # 7) LLM Re-ranking Stage (optional)
    llm_reranking_applied = False
    if use_llm_reranking and llm_client is not None:
        reranked = llm_rerank(llm_client, caption, top_labels, top_probs / top_probs.sum(), top_k=top_k)
        
        # Extract and normalize
        labels_r = [l for l, _ in reranked]
        probs_r = np.array([p for _, p in reranked], dtype=float)
        probs_r = probs_r / probs_r.sum()  # Ensure sum to 1
        
        llm_reranking_applied = True
        logger.info("LLM re-ranking applied")
    else:
        # Use original CLIP probabilities
        labels_r = top_labels
        probs_r = top_probs / top_probs.sum()
        logger.info("Using CLIP probabilities (no LLM re-ranking)")
    
    # 8) Get final top prediction
    top_label = labels_r[0]
    top_confidence = float(probs_r[0])
    
    # 9) LLM Explanation Stage: Generate UI reasoning
    reasoning = generate_ui_reasoning(
        llm_client,
        top_label,
        caption,
        attributes,
        domain_info['domain'],
        top_confidence
    )
    logger.info("Reasoning generated")
    
    # 10) Build final JSON response
    predictions = {
        label: float(prob * 100.0)
        for label, prob in zip(labels_r, probs_r)
    }
    
    result: Dict[str, Any] = {
        # Core predictions
        "predictions": predictions,
        "top_prediction": {
            "label": top_label,
            "score": float(probs_r[0] * 100.0)
        },
        "confidence_score": top_confidence,
        
        # Domain information
        "domain_info": domain_info,
        
        # UI-friendly reasoning
        "reasoning": reasoning,
        
        # Visual features
        "visual_features": attributes,
        
        # Alternative predictions
        "alternative_predictions": [
            {"label": label, "score": float(prob * 100.0)}
            for label, prob in zip(labels_r, probs_r)
        ],
        
        # Metadata
        "zero_shot": True,
        "multilingual_support": True,
        "language": "en",
        "temperature": temperature,
        "adaptive_module_used": adaptive_module is not None,
        "llm_reranking_used": llm_reranking_applied
    }
    
    logger.info(f"Pipeline complete: {top_label} ({top_confidence:.1%})")
    return result
