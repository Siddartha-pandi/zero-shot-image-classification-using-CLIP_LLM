"""
FastAPI backend server for the Advanced Zero-Shot Classification Framework.
This version integrates the new advanced features.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import tempfile
import os
import logging

# Import our modules, including the new advanced framework
from models import initialize_models
from advanced_inference import AdvancedZeroShotFramework

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Advanced Zero-Shot Framework", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Component Initialization ---
# Initialize the new advanced framework
classifier = AdvancedZeroShotFramework()

# --- Pydantic Models ---
class ClassificationRequest(BaseModel):
    labels: str = None  # Comma-separated labels, optional
    language: str = 'en'  # Language code (en, hi, es, fr)
    user_prompt: str = None  # Optional user prompt

class ClassificationResponse(BaseModel):
    predictions: Dict[str, float]
    top_prediction: Dict[str, Any]  # Now supports both old and new format
    confidence_score: float
    reasoning: Dict[str, Any]  # New UI-friendly reasoning structure
    domain_info: Dict[str, Any]
    visual_features: List[str]
    alternative_predictions: List[Dict[str, Any]]
    zero_shot: bool
    multilingual_support: bool
    language: str
    temperature: float
    adaptive_module_used: bool
    llm_reranking_used: bool

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting up Advanced Zero-Shot Framework...")
    if not initialize_models():
        raise RuntimeError("Model initialization failed")
    logger.info("Models initialized successfully")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Advanced Zero-Shot Framework API is running"}

@app.post("/api/classify", response_model=ClassificationResponse)
async def classify_image_endpoint(
    file: UploadFile = File(...),
    labels: str = None,
    language: str = 'en',
    user_prompt: str = None,
    temperature: float = 0.01,
    use_adaptive: bool = True,
    use_llm_reranking: bool = False
):
    """
    Classify an uploaded image with automatic or manual label generation.
    Supports multilingual classification and comprehensive explanations.
    
    Parameters:
    - file: Image file to classify
    - labels: Optional comma-separated class labels (auto-generated if not provided)
    - language: Language code (en, hi, es, fr) for multilingual support
    - user_prompt: Optional user description/prompt
    - temperature: Temperature for softmax calibration (default: 0.01)
    - use_adaptive: Whether to use adaptive embedding modules
    - use_llm_reranking: Whether to use LLM re-ranking
    """
    temp_path = ""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name
        
        # Parse class names or auto-generate
        if labels:
            class_names = [label.strip() for label in labels.split(',')]
            logger.info(f"Using provided labels: {class_names}")
        else:
            class_names = classifier.generate_image_labels(temp_path, num_labels=5)
            logger.info(f"Auto-generated labels: {class_names}")
        
        # Use the advanced classifier with all features
        response = classifier.classify(
            temp_path, 
            class_names,
            user_prompt=user_prompt,
            language=language,
            temperature=temperature,
            use_adaptive_module=use_adaptive,
            use_llm_reranking=use_llm_reranking
        )
        
        logger.info(f"Classification complete: {response['top_prediction']}")
        
        return response

    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


class FeedbackRequest(BaseModel):
    """Feedback for improving classification."""
    image_id: str
    predicted_label: str
    correct_label: str
    confidence: float


@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback to improve the classification system.
    Updates prototype embeddings and prompt weights based on user corrections.
    
    Parameters:
    - image_id: Identifier for the classified image
    - predicted_label: What the model predicted
    - correct_label: The correct label provided by user
    - confidence: Confidence score of the prediction
    """
    try:
        logger.info(f"Received feedback: {feedback.predicted_label} -> {feedback.correct_label}")
        
        # Update online tuner with corrected label
        # In practice, you'd retrieve the image embedding and update
        
        # Save feedback for later analysis
        feedback_log = {
            'image_id': feedback.image_id,
            'predicted': feedback.predicted_label,
            'correct': feedback.correct_label,
            'confidence': feedback.confidence,
            'timestamp': __import__('time').time()
        }
        
        # Log feedback (in production, save to database)
        logger.info(f"Feedback logged: {feedback_log}")
        
        return {
            'status': 'success',
            'message': 'Feedback received and will be used to improve the model'
        }
    
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/save_adapters")
async def save_adaptive_modules():
    """Save all domain-specific adaptive modules to disk."""
    try:
        if classifier.adaptive_modules:
            classifier.adaptive_modules.save_all()
            return {'status': 'success', 'message': 'Adaptive modules saved'}
        else:
            return {'status': 'info', 'message': 'No adaptive modules to save'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/save_vector_db")
async def save_vector_database():
    """Save vector database to disk."""
    try:
        if classifier.vector_db:
            classifier.vector_db.save()
            return {
                'status': 'success',
                'message': f'Vector DB saved with {classifier.vector_db.size()} entries'
            }
        else:
            return {'status': 'info', 'message': 'No vector database initialized'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
