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
class ClassificationResponse(BaseModel):
    predictions: Dict[str, float]
    top_prediction: Dict[str, float]
    narrative: str

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
    labels: str = "a cat,a dog,a car,a tree",
):
    """
    Classify an uploaded image using the new advanced zero-shot framework.
    """
    class_names = [label.strip() for label in labels.split(",") if label.strip()]
    if not class_names:
        raise HTTPException(status_code=400, detail="No valid labels provided")

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name
        
        # Use the new advanced classifier
        scores = classifier.classify(temp_path, class_names)
        
        # For demonstration, we'll generate a simple narrative.
        # In a full implementation, this would be more deeply integrated.
        narrative = classifier.model_manager.generate_narrative(f"The image likely contains one of the following: {', '.join(class_names)}.")

    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    if not scores:
        raise HTTPException(status_code=500, detail="Classification failed to produce scores.")

    top_class = max(scores, key=scores.get)
    top_prediction = {top_class: scores[top_class]}

    return {
        "predictions": scores,
        "top_prediction": top_prediction,
        "narrative": narrative,
    }
