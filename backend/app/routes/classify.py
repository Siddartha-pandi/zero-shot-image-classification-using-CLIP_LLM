# app/routes/classify.py
"""
Hybrid Classification Route
Main endpoint for ViT-H/14 + MedCLIP classification system
"""
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
import time
import logging
from typing import Optional, List

from ..models.router import get_router
from ..models.clip_vith14 import get_vith14_model
from ..models.medclip_model import get_medclip_model
from ..models.similarity import (
    compute_confidence_score,
    get_confidence_explanation,
    compute_cosine_similarity
)
from ..models.llm_hybrid import (
    generate_hybrid_explanation,
    generate_detailed_narrative,
    extract_objects_hybrid
)
from ..caption_service import generate_caption

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["classification"])

# Default class labels for different domains
DEFAULT_LABELS = {
    "medical": [
        "lungs",
        "rib cage",
        "pulmonary opacity",
        "heart",
        "chest",
        "normal chest x-ray",
        "pneumonia",
        "pleural effusion",
        "atelectasis",
        "cardiomegaly",
        "consolidation",
        "edema"
    ],
    "fashion": [
        "dress",
        "shirt",
        "pants",
        "jacket",
        "shoes",
        "accessories",
        "formal wear",
        "casual wear"
    ],
    "traffic": [
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "traffic sign",
        "road"
    ],
    "satellite": [
        "urban area",
        "vegetation",
        "water",
        "road network",
        "agricultural land",
        "forest",
        "buildings"
    ],
    "industrial": [
        "machinery",
        "factory",
        "assembly line",
        "warehouse",
        "industrial equipment",
        "manufacturing"
    ],
    "natural": [
        "dog",
        "cat",
        "bird",
        "tree",
        "flower",
        "person",
        "building",
        "landscape",
        "vehicle",
        "food"
    ]
}


@router.post("/classify-hybrid")
async def classify_hybrid(
    file: UploadFile = File(...),
    custom_labels: Optional[str] = Form(default=None),
    top_k: int = Form(default=5)
):
    """
    Hybrid classification endpoint with automatic model routing
    
    - Automatically detects domain using ViT-H/14
    - Routes to MedCLIP for medical images, ViT-H/14 for others
    - Generates LLM-based explanations
    - Returns comprehensive analysis
    """
    start_time = time.time()
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get router instance
        domain_router = get_router()
        
        # Parse custom labels or use defaults
        if custom_labels:
            labels = [label.strip() for label in custom_labels.split(",")]
        else:
            # Detect domain first for default labels
            _, domain, _, _ = domain_router.route(image)
            labels = DEFAULT_LABELS.get(domain, DEFAULT_LABELS["natural"])
        
        # Classify with automatic routing
        classification_result = domain_router.classify_with_routing(
            image=image,
            labels=labels,
            top_k=top_k
        )
        
        # Extract results
        model_used = classification_result["model_used"]
        domain = classification_result["domain"]
        domain_confidence = classification_result["domain_confidence"]
        domain_scores = classification_result["domain_scores"]
        predictions = classification_result["predictions"]
        image_emb = classification_result["image_embedding"]
        
        # Get top prediction
        top_prediction = predictions[0]
        prediction_label = top_prediction["label"]
        prediction_score = top_prediction["score"]
        
        # Generate caption
        caption = generate_caption(image)
        
        # Get appropriate model for caption embedding
        if model_used == "MedCLIP":
            model = get_medclip_model()
        else:
            model = get_vith14_model()
        
        # Encode caption for confidence calculation
        caption_emb = model.encode_text([caption])[0]
        
        # Compute final confidence score
        confidence_score = compute_confidence_score(
            domain=domain,
            model_used=model_used,
            prediction_score=prediction_score,
            image_emb=image_emb,
            caption_emb=caption_emb,
            domain_similarity=domain_confidence
        )
        
        # Generate LLM explanation
        llm_result = generate_hybrid_explanation(
            domain=domain,
            model_used=model_used,
            prediction=prediction_label,
            confidence=confidence_score,
            caption=caption,
            top_matches=predictions,
            domain_scores=domain_scores
        )
        
        # Generate detailed narrative
        narrative = generate_detailed_narrative(
            domain=domain,
            model_used=model_used,
            prediction=prediction_label,
            caption=llm_result["caption"],
            top_matches=predictions
        )
        
        # Extract objects
        objects = extract_objects_hybrid(
            domain=domain,
            caption=llm_result["caption"],
            top_matches=predictions
        )
        
        # Compute inference time
        inference_time = time.time() - start_time
        
        # Prepare response
        response = {
            "domain": domain,
            "model_used": model_used,
            "prediction": prediction_label,
            "confidence_score": round(confidence_score, 4),
            "caption": llm_result["caption"],
            "explanation": llm_result["explanation"],
            "risk_notes": llm_result.get("risk_notes", ""),
            "narrative": narrative,
            "objects": objects,
            "top_matches": [
                {
                    "label": pred["label"],
                    "score": round(pred["score"], 4)
                }
                for pred in predictions
            ],
            "domain_scores": {
                k: round(v, 4) for k, v in domain_scores.items()
            },
            "inference_time_seconds": round(inference_time, 3),
            "metadata": {
                "domain_confidence": round(domain_confidence, 4),
                "raw_prediction_score": round(prediction_score, 4),
                "total_labels_evaluated": len(labels),
                "confidence_explanation": get_confidence_explanation(
                    confidence_score, domain, model_used
                )
            }
        }
        
        logger.info(
            f"Classification complete: {prediction_label} ({confidence_score:.2%}) "
            f"using {model_used} in {inference_time:.2f}s"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "Internal server error during classification"}
        )


@router.get("/models/status")
async def get_models_status():
    """Get status of loaded models"""
    try:
        router_instance = get_router()
        
        return {
            "status": "ready",
            "models": {
                "vit_h14": {
                    "loaded": router_instance.vith14.model is not None,
                    "device": str(router_instance.vith14.device)
                },
                "medclip": {
                    "loaded": router_instance.medclip.model is not None,
                    "device": str(router_instance.medclip.device)
                }
            },
            "default_labels": DEFAULT_LABELS
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.get("/domains")
async def get_supported_domains():
    """Get list of supported domains"""
    return {
        "domains": list(DEFAULT_LABELS.keys()),
        "domain_descriptions": {
            "medical": "Medical imaging (X-rays, CT, MRI scans)",
            "fashion": "Fashion and apparel images",
            "traffic": "Traffic scenes and road images",
            "satellite": "Satellite and aerial imagery",
            "industrial": "Industrial and manufacturing scenes",
            "natural": "General natural and everyday scenes"
        }
    }
